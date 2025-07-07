using Pkg;
Pkg.activate(@__DIR__);
using Comonicon

using LinearAlgebra

include(joinpath(@__DIR__, "imaging_driver.jl"))
LinearAlgebra.BLAS.set_num_threads(1)
if Threads.nthreads() > 1
    VLBISkyModels.NFFT._use_threads[] = false
    VLBISkyModels.FFTW.set_num_threads(1)
end


"""
Generate random prior draws from the imaging model

# Arguments

 - `uvfile::String`: the path to the uvfits file. This is needed to set the beam size of the telescope.

# Options

- `-o, --outpath`: the path to the output directory where images and other stats will be saved. Default is the current directory.
- `--fovx`: the field of view in microarcseconds. Default is 200 μas.
- `--fovy`: the field of view in microarcseconds. Default is fovx.
- `-p, --psize`: the pixel size in microarcseconds. Default is 1 μas.
- `-x, --x`: the x offset of image center in microarcseconds. Default is 0 μas.
- `-y, --y`: the y offset of image center in microarcseconds. Default is 0 μas.
- `--pa`: The position angle of the grid in degrees. Default is 0 degrees.
- `--ftot`: The total flux. Can either we two numbers, i.e. 0.1, 2.5 which mean it fits the
            total flux within that range, or a single number which means it fixes the total flux
            using an apriori flux estimate.
- `-n, --nimgs`: the number of image posterior samples to generate. Default is 200.
- `--order`: the order of the Markov Random Field. Default is -1 which uses the Matern kernel.
- `--model`: The model to use for the prior image. Default is `ring` for others see below. 
- `--polrep`: The polarization representation. The default of PolExp which uses a matrix exponential representation.

# Flags

- `--scanavg`: Scan average the data prior to fitting. Note that if the data is merged multifrequency data, this will not work properly.
- `--space`: Flag space baselines. Namely this will flag any ground to space baselines.
- `--polarized`: Fit the polarized data.


# Notes
The details of the models are as follows:
 - `ring` is a ring with a constant background.
 - `ringnojet` is a ring with no constant background. 
 - `isojet` is a core with a constant floor that is fit. 
 - `jet` is fits the the jet with a asymmetric Gaussian. Note that `:jet` can be quite hard to fit. 
 - `flat` is a flat image with no structure.
- `--nogains`: Assumes that the instrument is perfect and does not have any gains. This is needed
  to decide whether the image centroid is fixed at the origin.

"""
@main function main(uvfile::String; outpath::String="",
    fovx::Float64=200.0, fovy::Float64=fovx,
    psize::Float64=1.0,
    x::Float64=0.0, y::Float64=0.0,
    pa::Float64=0.0,
    ftot::String="0.2, 2.5",
    nimgs::Int=200, 
    model::String="ring",
    scanavg::Bool=false,
    space::Bool=false,
    polarized::Bool=false,
    polrep::String="PolExp",
    order::Int=-1,
    nogains::Bool=false
)

    fovxrad = μas2rad(fovx)
    fovyrad = μas2rad(fovy)
    nx = ceil(Int, fovx / psize)
    ny = ceil(Int, fovy / psize)
    posang = deg2rad(pa)
    outpath = isempty(outpath) ? first(splitext(uvfile)) : joinpath(outpath, first(splitext(basename(uvfile))))
    @info "Estimating Beam from: $uvfile"
    @info "Outputing to $outpath"
    @info "Field of view: ($fovx, $fovy) μas"
    @info "number of pixels: ($nx, $ny)"
    @info "Image center offset: ($x, $y) μas"
    @info "PA of the grid $(pa) degrees"
    ftots = parse.(Float64, split(ftot, ","))
    if length(ftots) == 1
        @info "Using a fixed flux of $(ftots[1])"
        ftotpr = ftots[1]
    elseif length(ftots) == 2
        @info "Fitting the total flux between $(ftots[1]) and $(ftots[2])"
        ftotpr = Uniform(ftots[1], ftots[2])
    else
        throw(ArgumentError("The --ftot flag should have either one or two values while it parsed $(ftots)"))
    end

    if order < 0
        @info "Using Matern kernel for the stochastic model"
        base = Matern()
    else
        @info "Using Markov Random Field of order $order for the stochastic model"
        base = GMRF
    end
    if polarized
        @info "Using polarized model: $polrep"
        if polrep == "PolExp"
            prep = PolExp()
        elseif polrep == "Poincare"
            prep = Poincare()
        else
            throw(ArgumentError("Unknown polarized model: $polrep please pick from \"PolExp\", \"Poincare\""))
        end
    else
        @info "Only fitting the total intensity"
        prep = TotalIntensity()
    end

    obs = ehtim.obsdata.load_uvfits(uvfile)
    obs.add_scans()
    if scanavg
        obsavg = scan_average(obs)
    else
        obsavg = obs
    end
    if space
        @warn "We are flagging space baselines as requested by the `--space` flag"
        obsavg = obsavg.flag_sites(["space"])
    end
    data = extract_table(obsavg, Visibilities())
    x0 = μas2rad(x)
    y0 = μas2rad(y)
    hdr = ComradeBase.MinimalHeader(string(data.config.source),
        data.config.ra, data.config.dec,
        data.config.mjd, data[:baseline].Fr[1] # assume all frequencies are the same
    )
    if Threads.nthreads() > 1
        g = imagepixels(fovxrad, fovyrad, nx, ny, x0, y0; posang, executor=ThreadsEx(:dynamic), header=hdr)
    else
        g = imagepixels(fovxrad, fovyrad, nx, ny, x0, y0; posang, header=hdr)
    end
    beam = beamsize(data)
    @info "Beam relative to pixel size: = $(beam/μas2rad(psize))"
    if model == "ringnojet"
        mod = DblRing()
        @info "Assuming the image is a ring"
    elseif model == "ring"
        mod = DblRingWBkgd()
        @info "Assuming the image is a ring with a background jet"
    elseif model == "isojet"
        @info "Assuming the image is a isotropic jet structure"
        m = modify(Gaussian(), Stretch(beam / 2))
        mimg = intensitymap(m, g)
        mod = MimgPlusBkgd(mimg ./ sum(mimg))
    elseif model == "jet"
        @info "Assuming the image is an anisotropic jet structure"
        m = modify(Gaussian(), Stretch(beam / 2))
        mimg = intensitymap(m, g)
        mod = JetGauss(mimg ./ sum(mimg))
    elseif model == "lyapunov"
        @info "Assuming the image is a Lyanpunov ring structure"
        mod = LyapunovRing()
    elseif model == "flat"
        @info "No mean image"
        mod = Flat(g)
    else
        throw(ArgumentError("Unknown model: $model please pick from \"ringnojet\", \"ring\", \"isojet\", \"jet\""))
    end
    if !nogains
        imgmod = ImagingModel(prep, mod, g, ftotpr; base, order)
    else
        imgmod = ImagingModel(prep, mod, g, ftotpr; base, order, center=false)
    end
    skpr = skyprior(imgmod; beamsize=beam)
    skym = SkyModel(imgmod, skpr, g; algorithm=FINUFFTAlg(;threads=1))

    oskym, pr = Comrade.set_array(skym, arrayconfig(data))

    if pr isa Comrade.NamedDist
        npr = pr 
    else
        npr = Comrade.NamedDist(pr)
    end

    imgout = joinpath(mkpath(joinpath(dirname(outpath), "images")), basename(outpath)*"_results")
    @info "Saving images to $imgout"
    for i in 1:nimgs
        m = skymodel(oskym, rand(npr))
        img = intensitymap(m, g)
        Comrade.save_fits(imgout*"_draw$i.fits", img)
        p = imageviz(img)
        CairoMakie.save(imgout*"_draw$i.png", p)
    end
end
