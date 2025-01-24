abstract type AbstractImagingModel end

abstract type PolRep end
struct TotalIntensity <: PolRep end
struct Matern end

struct ImagingModel{P,M,G,F,B,C} <: AbstractImagingModel
    mimg::M
    grid::G
    ftot::F
    base::B
    order::Int
end

function ImagingModel(p::PolRep, mimg::M, grid, ftot; order=1, base=GMRF, center=centerfix(M)) where {M}
    b = prepare_base(base, grid, order)
    return ImagingModel{typeof(p),M,typeof(grid),typeof(ftot),typeof(b), center}(mimg, grid, ftot, b, order)
end

function ImagingModel(p::PolRep, mimg::IntensityMap, ftot; order=1, base=GMRF, center=centerfix(typeof(mimg)))
    return ImagingModel(p, mimg./sum(mimg), axisdims(mimg), ftot; order=order, base=base, center)
end


prepare_base(b::Type{<:VLBIImagePriors.MarkovRandomField}, grid, order) = b
prepare_base(::Matern, grid, order) = first(matern(size(grid)))



center(::ImagingModel{P, M, G, F, B, C}) where {P, M, G, F, B, C} = C

centerfix(::Type{<:Any}) = true

getftot(m::ImagingModel{P, M, G, <:Real}, _)   where {P, M, G} = m.ftot
getftot(::ImagingModel{P, M, G}, θ) where {P, M, G} = θ.ftot

function (m::ImagingModel{P})(θ, meta) where {P}
    mimg = make_mean(m.mimg, RectiGrid(dims(m.grid)[1:2]), θ)
    fimg = getftot(m, θ)

    pmap = make_image(P, m.base, fimg, mimg, θ)
    if center(m)
        x0, y0 = centroid(pmap)
        return shifted(ContinuousImage(pmap, BSplinePulse{3}()), -x0, -y0)
    else
        return ContinuousImage(pmap, BSplinePulse{3}())
    end

end

function make_image(::Type{<:TotalIntensity}, ::Type{<:VLBIImagePriors.MarkovRandomField}, ftot, mimg, θ)
    (;c, σ) = θ
    img = apply_fluctuations(CenteredLR(), mimg, σ*c.params)
    bimg = baseimage(img)
    for i in eachindex(img)
        bimg[i] *= ftot
    end
    return  img
end

function make_image(::Type{<:TotalIntensity}, trf::VLBIImagePriors.StationaryMatern, ftot, mimg, θ)
    (;c, σ, ρ, ν) = θ
    δ = trf(c, ρ, ν)
    δ .*= σ
    img = apply_fluctuations(CenteredLR(), mimg, δ)
    bimg = baseimage(img)
    for i in eachindex(img)
        bimg[i] *= ftot
    end
    return img
end



function skyprior(m::ImagingModel{P}; beamsize=μas2rad(20.0), overrides::Dict=Dict()) where {P}
    imgprior = genimgprior(P, m.base, m.grid, beamsize, m.order)
    mprior = genmeanprior(m.mimg)

    if !(m.ftot isa Real)
        imgprior[:ftot] = m.ftot
    end

    prior = merge(imgprior, mprior)

    for k in keys(overrides)
        prior[k] = overrides[k]
    end


    return NamedTuple(prior)
end

function genimgprior(::Type{<:TotalIntensity}, base::Type{<:VLBIImagePriors.MarkovRandomField}, grid, beamsize, order)
    cprior = corr_image_prior(grid, beamsize; base=base, order=order, lower=5.0)
    default = Dict(
        :c => cprior,
        :σ => truncated(Normal(0.0, 0.1); lower = 0.0)
        )
    return default
end

function genimgprior(::Type{<:TotalIntensity}, base::VLBIImagePriors.StationaryMatern, grid, beamsize, order)
    bs = beamsize/pixelsizes(grid).X
    cprior = VLBIImagePriors.std_dist(base)
    default = Dict(
        :c => cprior,
        :σ => truncated(Normal(0.0, 0.5); lower = 0.0),
        :ρ => 0.5+InverseGamma(1.0, -log(0.1)*bs),
        :ν => 0.1+InverseGamma(5.0, 9.0)
        )
    return default
end


function make_mean(mimg::IntensityMap, grid, θ)
    return mimg
end

function genmeanprior(::IntensityMap)
    return Dict()
end

struct MimgPlusBkgd{M}
    mimg::M
end

function make_mean(mimg::MimgPlusBkgd, grid, θ)
    (;fb) = θ
    fbn = fb/(prod(size(grid)))
    return mimg.mimg.*((1-fb)) .+ fbn
end

function genmeanprior(::MimgPlusBkgd)
    return Dict(:fb => Beta(1.0, 5.0))
end

struct JetGauss{M}
    core::M
end

function make_mean(mimg::JetGauss, grid, θ)
    (;r, τ, ξτ, x, y, fj) = θ
    img = intensitymap(modify(Gaussian(), Stretch(r, r*(1+τ)), Rotate(ξτ/2), Shift(x, y)), grid)
    fl = sum(img)
    img .= mimg.core .* (1-fj) .+ img./fl .* fj
    return img
end

function genmeanprior(m::JetGauss)
    fovx, fovy = fieldofview(m.core)
    dx, dy = pixelsizes(m.core)
    return Dict(
        :r  => Uniform(dx*10, fovx/3),
        :τ  => Exponential(1.0),
        :ξτ => DiagonalVonMises(0.0, inv(π^2)),
        :x  => Uniform(-fovx/4, fovx/4),
        :y  => Uniform(-fovy/4, fovy/4),
        :fj => Beta(1.0, 5.0)
        )
end


struct DblRing end
centerfix(::Type{<:DblRing}) = false

function make_mean(::DblRing, grid, θ)
    (;r0, ain, aout,) = θ
    m = modify(RingTemplate(RadialDblPower(ain, aout), AzimuthalUniform()), Stretch(r0))
    mimg = intensitymap(m, grid)
    pmimg = baseimage(mimg)
    pmimg .= pmimg./sum(pmimg)
    return mimg 
end

function genmeanprior(::DblRing)
    return Dict(
        :r0        => Uniform(μas2rad(15.0), μas2rad(35.0)),
        :ain       => Exponential(1.0)+1,
        :aout      => Exponential(1.0)+1
        )
end


struct DblRingWBkgd end
centerfix(::Type{<:DblRingWBkgd}) = false

function make_mean(::DblRingWBkgd, grid, θ)
    (;r0, ain, aout, fb) = θ
    fbn = fb/(prod(size(grid)))
    m = modify(RingTemplate(RadialDblPower(ain, aout), AzimuthalUniform()), Stretch(r0))
    mimg = intensitymap(m, grid)
    pmimg = baseimage(mimg)
    pmimg .= pmimg./sum(pmimg).*(1-fb) .+ fbn
    return mimg 
end

function genmeanprior(::DblRingWBkgd)
    return Dict(
        :r0        => Uniform(μas2rad(15.0), μas2rad(35.0)),
        :ain       => Exponential(3.0),
        :aout      => Exponential(3.0)+1,
        :fb        => Beta(1.0, 5.0)
        )
end