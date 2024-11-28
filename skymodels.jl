
@kwdef struct RingMaternImage{G,M,D}
    grid::G
    trf::M
    dstd::D
    function RingMaternImage(grid)
        trf, dstd = matern(size(grid))
        new{typeof(grid), typeof(trf), typeof(dstd)}(grid, trf, dstd)
    end
end

Enzyme.EnzymeRules.inactive_type(::Type{<:RingMaternImage}) = true



function (gmrf::RingMaternImage)(θ, meta)
    (; c, ρ, ν, σ, ftot, r, αin, αout) = θ
    m = modify(RingTemplate(RadialDblPower(αin, αout), AzimuthalUniform()), Stretch(r))
    mimg = intensitymap(m, gmrf.grid)
    pmimg = baseimage(mimg)
    s = Comrade._fastsum(pmimg)
    pmimg = baseimage(mimg)
    pmimg ./= s
    rast = _make_image(mimg, ftot, σ, gmrf.trf(c, ρ, ν))
    m = ContinuousImage(rast, DeltaPulse())
    return m
end

function skyprior(m::RingMaternImage; beamsize=μas2rad(20.0), overrides::Dict=Dict())
    bs = beamsize/step(m.grid.X)
    default = Dict(
        :ftot => Uniform(0.2, 2.5),
        :c    => m.dstd,
        :ρ    => 0.5+InverseGamma(1.0, -log(0.1)*bs),
        :ν    => 0.1+InverseGamma(5.0, 9.0), 
        :αin  => Uniform(1.0, 10.0),
        :αout => Uniform(1.0, 10.0),
        :r    => Uniform(μas2rad(10.0), μas2rad(30.0)),
        :σ    => truncated(Normal(0.0, 0.5); lower = 0.0, upper=5.0),
    )

    for k in keys(overrides)
        default[k] = overrides[k]
    end

    return NamedTuple(default)
end

struct JetMaternImage{I,G,M,D}
    mimg::I
    grid::G
    trf::M
    dstd::D
    function JetMaternImage(mimg::IntensityMap)
        simg = mimg./sum(mimg)
        grid = axisdims(mimg)
        trf, dstd = matern(size(grid))  
        new{typeof(mimg), typeof(grid), typeof(trf), typeof(dstd)}(simg, grid, trf, dstd)
    end
end
Enzyme.EnzymeRules.inactive_type(::Type{<:JetMaternImage}) = true

function _make_image(mimg::IntensityMap, ftot, σ, δ)
    out = similar(δ)
    # We do this for numerical stability
    δmax = maximum(δ)
    @inbounds for i in eachindex(out, δ)
        out[i] = σ*(δ[i] - δmax)
    end
    rast = apply_fluctuations(CenteredLR(), mimg, out)
    prast = baseimage(rast)
    prast .*= ftot
    return rast
end

function (gmrf::JetMaternImage)(θ, meta)
    (; ftot, c, σ, ρ, ν, fb) = θ
    fbn = fb/length(gmrf.mimg)
    mimg = similar(gmrf.mimg)
    mimg = (gmrf.mimg .+ fbn)./(1 + fb)
    rast = _make_image(mimg, ftot, σ, gmrf.trf(c, ρ, ν))
    x0, y0 = centroid(rast)
    return modify(ContinuousImage(rast, DeltaPulse()), Shift(-x0, -y0))
end

function skyprior(m::JetMaternImage; beamsize=μas2rad(20.0), overrides::Dict=Dict())
    bs = beamsize/step(m.grid.X)
    default = Dict(
        :ftot => Uniform(0.2, 2.5),
        :c    => m.dstd,
        :σ    => truncated(Normal(0.0, 0.5); lower = 0.0, upper=5.0),
        :ρ    => InverseGamma(1.0, -log(0.1)*bs),
        :ν    => InverseGamma(5.0, 9.0), 
        :fb   => Exponential(0.1),
    )

    for k in keys(overrides)
        default[k] = overrides[k]
    end

    return NamedTuple(default)
end



@kwdef struct JohnsonSUMaternImage{G,M,D}
    grid::G
    trf::M
    dstd::D
    function JohnsonSUMaternImage(grid)
        trf, dstd = matern(size(grid))
        new{typeof(grid), typeof(trf), typeof(dstd)}(grid, trf, dstd)
    end
end

function (gmrf::JohnsonSUMaternImage)(θ, meta)
    (;ftot, c, σ, ρ, ν, r, σring, γ) = θ
    m = modify(RingTemplate(RadialJohnsonSU(σring/r, γ), AzimuthalUniform()), Stretch(r))
    mimg = intensitymap(m, gmrf.grid)
    pmimg = baseimage(mimg)
    s = Comrade._fastsum(pmimg)
    pmimg = baseimage(mimg)
    pmimg ./= s
    rast = _make_image(mimg, ftot, σ, gmrf.trf(c, ρ, ν))
    m = ContinuousImage(rast, DeltaPulse())
end

function skyprior(m::JohnsonSUMaternImage; beamsize=μas2rad(20.0), overrides::Dict=Dict())
    bs = beamsize/step(m.grid.X)
    default = Dict(
        :ftot => Uniform(0.2, 2.5),
        :c    => m.dstd,
        :ρ    => 0.5+InverseGamma(1.0, -log(0.1)*bs),
        :ν    => 0.1+InverseGamma(5.0, 9.0), 
        :σring=> Uniform(μas2rad(1.0), μas2rad(30.0)),
        :γ    => Normal(0.0, 2.0),
        :r    => Uniform(μas2rad(10.0), μas2rad(40.0)),
        :σ    => truncated(Normal(0.0, 0.5); lower = 0.0, upper=5.0),
    )

    for k in keys(overrides)
        default[k] = overrides[k]
    end

    return NamedTuple(default)
end
