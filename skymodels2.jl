using StatsFuns: logistic
abstract type PolRep end
abstract type PolModel <: PolRep end
struct Poincare <: PolModel end
struct PolExp <: PolModel end
struct TotalIntensity <: PolRep end
struct Matern end

struct ImagingModel{P,M,G,F,B,C}
    mimg::M
    grid::G
    ftot::F
    base::B
    order::Int
end
Enzyme.EnzymeRules.inactive_type(::Type{<:ImagingModel}) = true


function fast_centroid(img::IntensityMap{<:Real,2})
    x0 = zero(eltype(img))
    y0 = zero(eltype(img))
    dp = domainpoints(img)
    fs = Comrade._fastsum(img)
    @inbounds for i in CartesianIndices(img)
        x0 += dp[i].X*img[i]
        y0 += dp[i].Y*img[i]
    end
    return x0/fs, y0/fs
end

fast_centroid(img::IntensityMap{<:StokesParams}) = fast_centroid(stokes(img, :I))


function ImagingModel(p::PolRep, mimg::M, grid, ftot; order=1, base=GMRF, center=centerfix(M)) where {M}
    b = prepare_base(base, grid, order)
    bt = isa(base, Matern) ? typeof(b) : Type{base}
    return ImagingModel{typeof(p),M,typeof(grid),typeof(ftot),bt,center}(mimg, grid, ftot, b, order)
end

@inline prepare_base(b::Type{<:VLBIImagePriors.MarkovRandomField}, grid, order) = b
@inline prepare_base(::Matern, grid, order) = first(matern(size(grid); executor=(Threads.nthreads() > 1 ? :dynamic : :serial)))

function ImagingModel(p::PolRep, mimg::IntensityMap, ftot; order=1, base=GMRF)
    return ImagingModel(p, mimg./sum(mimg), axisdims(mimg), ftot; order=order, base=base)
end

@inline center(::ImagingModel{P, M, G, F, B, C}) where {P, M, G, F, B, C} = C

getftot(m::ImagingModel{P, M, G, <:Real}, _)   where {P, M, G} = m.ftot
getftot(::ImagingModel{P, M, G}, θ) where {P, M, G} = θ.ftot

function (m::ImagingModel{P})(θ, meta) where {P}
    mimg = make_mean(m.mimg, m.grid, θ)
    fimg = getftot(m, θ)
    
    pmap = make_image(P, m.base, fimg, mimg, θ)
    if center(m)
        x0, y0 = fast_centroid(pmap)
        ms = shifted(ContinuousImage(pmap, BSplinePulse{3}()), -x0, -y0)
    else
        ms = ContinuousImage(pmap, BSplinePulse{3}())
    end

    return ms
end

@inline function make_image(::Type{<:Poincare}, ::Type{<:VLBIImagePriors.MarkovRandomField}, ftot, mimg, θ)
    (;c, σ, p, p0, pσ, angparams) = θ
    return make_poincare(ftot, mimg, σ.*c.params, p0, pσ, p.params, angparams)
end

@inline function make_image(::Type{<:PolExp}, ::Type{<:VLBIImagePriors.MarkovRandomField}, ftot, mimg, θ)
    (;a, b, c, d, σa, σb, σc, σd,) = θ
    δa = similar(a.params)
    δb = similar(b.params)
    δc = similar(c.params)
    δd = similar(d.params)
    @inbounds for i in eachindex(δa, δb, δc, δd)
        δa[i] = σa*a.params[i]
        δb[i] = σb*b.params[i]
        δc[i] = σc*c.params[i]
        δd[i] = σd*d.params[i]
    end
    return make_pol2expimage(ftot, δa, δb, δc, δd, mimg)
end

@inline function make_image(::Type{<:Poincare}, trf::VLBIImagePriors.StationaryMatern, ftot, mimg, θ)
    (;c, σ, ρ, ν, p, p0, pσ, pν, pρ, angparams) = θ
    δ = trf(c, ρ, ν)
    pδ= trf(p, pρ, pν)
    for i in eachindex(δ)
        δ[i] *= σ
    end
    return make_poincare(ftot, mimg, δ, p0, pσ, pδ, angparams)
end

@inline function make_image(::Type{<:PolExp}, trf::VLBIImagePriors.StationaryMatern, ftot, mimg, θ)
    (;a, b, c, d, ρa, ρb, ρc, ρd, νa, νb, νc, νd, σa, σb, σc, σd,) = θ
    δa = trf(a, ρa, νa)
    δb = trf(b, ρb, νb)
    δc = trf(c, ρc, νc)
    δd = trf(d, ρd, νd)
    @inbounds for i in eachindex(δa, δb, δc, δd)
        δa[i] *= σa
        δb[i] *= σb
        δc[i] *= σc
        δd[i] *= σd
    end 
    return make_pol2expimage(ftot, δa, δb, δc, δd, mimg)
end

@inline function make_image(::Type{<:TotalIntensity}, trf::VLBIImagePriors.StationaryMatern, ftot, mimg, θ)
    (;c, σ, ρx, ρy, ρξ, ν) = θ
    δ = trf(c, (ρx, ρy), ρξ/2, ν)
    for i in eachindex(δ)
        δ[i] *= σ
    end
    return make_stokesi(ftot, mimg, δ)
end

@inline function make_image(::Type{<:TotalIntensity}, ::Type{<:VLBIImagePriors.MarkovRandomField}, ftot, mimg, θ) 
    make_stokesi(ftot, mimg, θ.σ.*θ.c.params)
end

@inline function make_stokesi(ftot, mimg, δ)
    stokesi = apply_fluctuations(CenteredLR(), mimg, δ)
    pstokesi = baseimage(stokesi)
    for i in eachindex(pstokesi)
        pstokesi[i] *= ftot
    end
    return stokesi
end



function make_poincare(ftot, mimg, δ, p0, pσ, pδ, angparams)
    stokesi = apply_fluctuations(CenteredLR(), mimg, δ)
    pstokesi = parent(stokesi)
    for i in eachindex(pstokesi)
        pstokesi[i] *= ftot
    end
    ptotim  = logistic.(p0 .+ pσ.*pδ)
    pmap = PoincareSphere2Map(stokesi, ptotim, angparams)
    return pmap
end

function make_pol2expimage(ftot, a, b, c, d, mimg)
    # this allocated a whole new map so we can do things in place after
    δ = VLBISkyModels.PolExp2Map!(a, b, c, d, axisdims(mimg))
    brast = baseimage(δ)
    δI= Comrade._fastsum(brast.I)
    fr = zero(δI)
    @inbounds for i in eachindex(mimg, brast)
        brast[i] *= mimg[i]/δI
        fr += brast[i].I
    end

    for i in eachindex(brast)
        brast[i] *= ftot/fr
    end
    pmap = IntensityMap(brast, axisdims(mimg)) 
    return pmap
end

centerfix(::Type{<:Any}) = true

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

function genimgprior(::Type{<:Poincare}, base::Type{<:VLBIImagePriors.MarkovRandomField}, grid, beamsize, order)
    cprior = corr_image_prior(grid, beamsize; base=base, order=order, lower=4.0)
    default = Dict(
        :c => cprior,
        :σ => truncated(Normal(0.0, 0.5); lower = 0.0),
        :p => cprior,
        :p0=> Normal(-1.0, 2.0),
        :pσ=> truncated(Normal(0.0, 0.5); lower = 0.0),
        :angparams => ImageSphericalUniform(size(cprior.priormap.cache)...)
        )
    return default
end

function genimgprior(::Type{<:PolExp}, base::Type{<:VLBIImagePriors.MarkovRandomField}, grid, beamsize, order)
    cprior = corr_image_prior(grid, beamsize; base=base, order=order, lower=4.0)
    default = Dict(
        :a => cprior,
        :b => cprior,
        :c => cprior,
        :d => cprior,
        :σa => truncated(Normal(0.0, 0.5); lower=0.0),
        :σb => truncated(Normal(0.0, 0.5); lower=0.0),
        :σc => truncated(Normal(0.0, 0.5); lower=0.0),
        :σd => truncated(Normal(0.0, 0.05); lower=0.0),
        )
    return default
end

function genimgprior(::Type{<:Poincare}, base::VLBIImagePriors.StationaryMatern, grid, beamsize, order)
    bs = beamsize/step(grid.XL)
    cprior = VLBIImagePriors.std_dist(base)
    ρpr = truncated(InverseGamma(1.0, -log(0.1)*bs); lower = 4.0, upper=2*max(size(grid)...))
    νpr = truncated(InverseGamma(5.0, 9.0); lower = 0.1)

    default = Dict(
        :c  => cprior,
        :σ  => truncated(Normal(0.0, 0.5); lower = 0.0),
        :ρ  => ρpr,
        :ν  => νpr,
        :p  => cprior,
        :ρp => ρpr,
        :νp => νpr, 
        :p0 => Normal(-1.0, 2.0),
        :pσ => truncated(Normal(0.0, 0.5); lower = 0.0),
        :angparams => ImageSphericalUniform(size(cprior.priormap.cache)...)
        )
    return default
end

function genimgprior(::Type{<:PolExp}, base::VLBIImagePriors.StationaryMatern, grid, beamsize, order)
    bs = beamsize/step(grid.X)
    cprior = VLBIImagePriors.std_dist(base)
    ρpr = truncated(InverseGamma(1.0, -log(0.1)*bs); lower = 4.0, upper=2*max(size(grid)...))
    νpr = truncated(InverseGamma(5.0, 9.0); lower = 0.1)

    default = Dict(
        :a => cprior,
        :b => cprior,
        :c => cprior,
        :d => cprior,
        :σa => truncated(Normal(0.0, 1.0); lower=0.0),
        :σb => truncated(Normal(0.0, 0.5); lower=0.0),
        :σc => truncated(Normal(0.0, 0.5); lower=0.0),
        :σd => truncated(Normal(0.0, 0.1); lower=0.0),
        :ρa    => ρpr,
        :νa    => νpr, 
        :ρb    => ρpr,
        :νb    => νpr,
        :ρc    => ρpr,
        :νc    => νpr,
        :ρd    => ρpr,
        :νd    => νpr
        )
    return default
end

function genimgprior(::Type{<:TotalIntensity}, base::Type{<:VLBIImagePriors.MarkovRandomField}, grid, beamsize, order)
    cprior = corr_image_prior(grid, beamsize; base=base, order=order, lower=4.0)
    default = Dict(
        :c => cprior,
        :σ => truncated(Normal(0.0, 0.5); lower = 0.0)
        )
    return default
end

function genimgprior(::Type{<:TotalIntensity}, base::VLBIImagePriors.StationaryMatern, grid, beamsize, order)
    bs = beamsize/step(grid.X)
    cprior = VLBIImagePriors.std_dist(base)
    ρpr = truncated(InverseGamma(1.0, -log(0.1)*bs); lower = 4.0, upper=2*max(size(grid)...))
    νpr = truncated(InverseGamma(5.0, 9.0); lower = 0.1)

    default = Dict(
        :c => cprior,
        :σ => truncated(Normal(0.0, 1.0); lower = 0.0),
        :ρx => ρpr,
        :ρy => ρpr,
        :ρξ => DiagonalVonMises(0.0, inv(π^2)),
        :ν => νpr
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
    (; fb) = θ
    fbn = fb / (prod(size(grid)))
    return mimg.mimg .* ((1 - fb)) .+ fbn
end

function genmeanprior(::MimgPlusBkgd)
    return Dict(:fb => Beta(1.0, 5.0))
end

struct JetGauss{M}
    core::M
end

function make_mean(mimg::JetGauss, grid, θ)
    (; r, τ, ξτ, x, y, fj) = θ
    img = intensitymap(modify(Gaussian(), Stretch(r, r * (1 + τ)), Rotate(ξτ / 2), Shift(x, y)), grid)
    fl = Comrade._fastsum(img)
    pimg = baseimage(img)
    pcore = baseimage(mimg.core)
    @inbounds for i in eachindex(pimg, pcore)
        pimg[i] = pcore[i] * (1 - fj) + pimg[i] / fl * fj
    end
    return img
end

function genmeanprior(m::JetGauss)
    fovx, fovy = fieldofview(m.core)
    x0, y0 = phasecenter(m.core)
    dx, dy = pixelsizes(m.core)
    return Dict(
      :r => Uniform(dx * 4, fovx / 3),
      :τ => Uniform(0.0, 10.0),
      :ξτ => DiagonalVonMises(0.0, inv(π^2)),
      :x => Uniform(-fovx / 4 - x0, fovx / 4 - x0),
      :y => Uniform(-fovy / 4 - y0, fovy / 4 - y0),
      :fj => Beta(1.0, 5.0)
    )
end


struct DblRing end
centerfix(::Type{<:DblRing}) = false

function make_mean(::DblRing, grid, θ)
    (; r0, ain, aout,) = θ
    m = modify(RingTemplate(RadialDblPower(ain, aout), AzimuthalUniform()), Stretch(r0))
    mimg = intensitymap(m, grid)
    pmimg = baseimage(mimg)
    pmimg .*= inv(Comrade._fastsum(pmimg))
    return mimg
end

function genmeanprior(::DblRing)
    return Dict(
      :r0 => Uniform(μas2rad(10.0), μas2rad(40.0)),
      :ain => Uniform(0.0, 6.0),
      :aout => Uniform(1.0, 6.0),
    )
end


struct DblRingWBkgd end
centerfix(::Type{<:DblRingWBkgd}) = false

function make_mean(::DblRingWBkgd, grid, θ)
    (; r0, ain, aout, fb) = θ
    fbn = fb / (prod(size(grid)))
    m = modify(RingTemplate(RadialDblPower(ain, aout), AzimuthalUniform()), Stretch(r0))
    mimg = intensitymap(m, grid)
    pmimg = baseimage(mimg)
    f = Comrade._fastsum(pmimg)
    @inbounds for i in eachindex(pmimg)
        pmimg[i] = pmimg[i] / f * (1 - fb) + fbn
    end
    return mimg
end

function genmeanprior(::DblRingWBkgd)
    return Dict(
      :r0 => Uniform(μas2rad(10.0), μas2rad(40.0)),
      :ain => Uniform(0.0, 6.0),
      :aout => Uniform(1.0, 6.0),
      :fb => Beta(1.0, 5.0)
    )
end


struct LyapunovRing end
centerfix(::Type{<:LyapunovRing}) = false


function make_mean(::LyapunovRing, grid, θ)
    (; r0, w, α0, r1, γ, α1, f1, x1, y1) = θ

    m0 = modify(RingTemplate(RadialJohnsonSU(w, α0), AzimuthalUniform()), Stretch(r0))
    m1 = modify(RingTemplate(RadialJohnsonSU(w*exp(-γ), α1), AzimuthalUniform()), 
                Stretch(r1), Shift(x1, y1))

    mimg = intensitymap(m0 + f1*m1, grid)

    pmimg = baseimage(mimg)
    f = Comrade._fastsum(pmimg)
    @inbounds for i in eachindex(pmimg)
        pmimg[i] = pmimg[i] / f
    end
    return mimg
end

function genmeanprior(::LyapunovRing)
    return Dict(
      :r0 => Uniform(μas2rad(10.0), μas2rad(40.0)),
      :w  => Uniform(0.0, 1.0),
      :α0 => Normal(),
      :r1 => Uniform(μas2rad(10.0), μas2rad(40.0)),
      :γ   => Uniform(0.0, π),
      :α1 => Normal(),
      :f1 => Uniform(0.0, 1.0),
      :x1 => Uniform(-μas2rad(10.0), μas2rad(10.0)),
      :y1 => Uniform(-μas2rad(10.0), μas2rad(10.0))
    )
end
