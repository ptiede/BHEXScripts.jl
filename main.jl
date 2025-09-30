using Pkg;
Pkg.activate(@__DIR__);
using Comonicon

using LinearAlgebra
using NonuniformFFTs

include(joinpath(@__DIR__, "imaging_driver.jl"))
LinearAlgebra.BLAS.set_num_threads(1)
if Threads.nthreads() > 1
    VLBISkyModels.NFFT._use_threads[] = false
    VLBISkyModels.FFTW.set_num_threads(Threads.nthreads())
end


"""
Fits BHEX data using Comrade and ring prior for the image.

# Arguments

 - `uvfile::String`: the path to the uvfits file you want to fit

# Options

- `-o, --outpath`: the path to the output directory where images and other stats will be saved. Default is the current directory.
- `-a, --array`: the ehtim array or tarr file. If `--polarized` is used then this must be specified.
- `--fovx`: the field of view in microarcseconds. Default is 200 μas.
- `--fovy`: the field of view in microarcseconds. Default is fovx.
- `-p, --psize`: the pixel size in microarcseconds. Default is 1 μas.
- `-x, --x`: the x offset of image center in microarcseconds. Default is 0 μas.
- `-y, --y`: the y offset of image center in microarcseconds. Default is 0 μas.
- `--pa`: The position angle of the grid in degrees. Default is 0 degrees.
- `--ftot`: The total flux. Can either we two numbers, i.e. 0.1, 2.5 which mean it fits the
            total flux within that range, or a single number which means it fixes the total flux
            using an apriori flux estimate.
- `-u, --uvmin`: the minimum uv distance in λ. Default is 0.0e9.
- `-n, --nimgs`: the number of image posterior samples to generate. Default is 200.
- `--lg`: the log-gain amplitude prior standard deviation. Default is 0.2.
- `--nsample`: the number of MCMC samples from the posterior. Default is 5_000.
- `--nadapt`: the number of MCMC samples to use for adaptation. Default is 2_500.
- `-f, --ferr`: the fractional error in the data. Default is 0.0.
- `--order`: the order of the Markov Random Field. Default is -1 which uses the Matern kernel. 0 means that we fit a MRF with order 1,2,3.
- `--model`: The model to use for the prior image. Default is `ring` for others see below. 
- `--maxiters`: the maximum number of iterations for the optimizer. Default is 15_000.
- `--ntrials`: the number of trials to run the optimizer. Default is 10.
- `--polrep`: The polarization representation. The default of PolExp which uses a matrix exponential representation.
- `--refsite`: The reference site for EVPA calibration. Default is `ALMA`.
- `--fthreads`: The number of threads to use for the FINUFFT algorithm. Default is 1. For large data sets make this bigger.

# Flags

- `-r, ---restart`: Restart a previous checkpointed run assuming the checkpoint file is in the outpath.
- `-b, --benchmark`: Run a benchmarking test to see how long it takes to evaluate the logdensity and its gradient.
- `--scanavg`: Scan average the data prior to fitting. Note that if the data is merged multifrequency data, this will not work properly.
- `--flgspace`: Flag space baselines. Namely this will flag any ground to space baselines.
- `--polarized`: Fit the polarized data. This requires the `--array` flag to be set.
- `--frcal`: Flag that the data has been FR-cal'd (not the default in ngehtsim)
- `--noleakage`: Assumes that the data doesn't have leakage.
- `--nogains`: Assumes that the instrument is perfect and does not have any gains. 
- `--gauto`: Use the automatic gain model that fits for gain offsets and dispersion.


# Notes
The details of the models are as follows:
 - `ring` is a ring with a constant background.
 - `ringnojet` is a ring with no constant background. 
 - `isojet` is a core with a constant floor that is fit. 
 - `jet` is fits the the jet with a asymmetric Gaussian. Note that `:jet` can be quite hard to fit. 
 - `flat` is a flat image with no structure.

"""
@main function main(uvfile::String; outpath::String="",
    array::String="",
    fovx::Float64=200.0, fovy::Float64=fovx,
    psize::Float64=1.0,
    x::Float64=0.0, y::Float64=0.0,
    pa::Float64=0.0,
    ftot::String="0.2, 2.5",
    uvmin::Float64=0e9,
    nimgs::Int=200, lg::Float64=0.2,
    model::String="ring",
    restart::Bool=false, benchmark::Bool=false, nsample::Int=5_000, nadapt::Int=2_500,
    scanavg::Bool=false,
    space::Bool=false,
    ferr::Float64=0.0,
    maxiters::Int=15_000,
    polarized::Bool=false,
    polrep::String="PolExp",
    frcal::Bool=false,
    ntrials::Int=10,
    noleakage::Bool=false,
    nogains::Bool=false,
    gauto::Bool=false,
    order::Int=-1,
    fthreads::Int=1
)

    fovxrad = μas2rad(fovx)
    fovyrad = μas2rad(fovy)
    nx = ceil(Int, fovx / psize)
    ny = ceil(Int, fovy / psize)
    posang = deg2rad(pa)
    outpath = isempty(outpath) ? first(splitext(uvfile)) : joinpath(outpath, first(splitext(basename(uvfile))))
    @info "Fitting the data: $uvfile"
    @info "Loading the array file: $array"
    @info "Outputing to $outpath"
    @info "Field of view: ($fovx, $fovy) μas"
    @info "number of pixels: ($nx, $ny)"
    @info "Image center offset: ($x, $y) μas"
    @info "PA of the grid $(pa) degrees"
    @info "Adding $ferr fractional error to the data"
    @info "Is the data frcal'd: $frcal"
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

    if polarized && noleakage
        @warn("I am assuming that you want polarized fits but are not fitting leakage.\n" *
              "This means I am not going to include feed rotations or gain ratios in the model\n" *
              "Open an issue if this is not what you want")
    end

    if nsample <= 5000 && polarized
        @warn "5000 samples for polarized imaging is likely not enough. Please at least\n" *
              "double this and the number of adaptation samples if you want a well sampled posterior."
    end

    if order < 0
        @info "Using Matern kernel with circular b.c. for the stochastic model"
        base = Matern()
    elseif order == 0
        @info "Using Markov Random Field expansion with circular b.c. of order 3 for the stochastic model"
        base = MarkovRF(3)
    else
        @info "Using Markov Random Field with Dirichlet b.c. of order $order for the stochastic model"
        base = GMRF
    end

    if base isa Matern || base isa MarkovRF || (base === GMRF && order == 1) 
        nx2 = nextprod((2,3,5,7), nx)
        ny2 = nextprod((2,3,5,7), ny)
        if (nx2 != nx) || (ny2 != ny)
            @warn "You are using a $base stochastic model with an image size of ($nx, $ny) which is not optimal for perf.\n" *
                  "I am changing the image size to ($nx2, $ny2) which is optimal for $base"
            nx = nx2
            ny = ny2
        end
    elseif base === GMRF && order > 1
        nx2 = nextprod((2,3,5,7), nx+1)
        ny2 = nextprod((2,3,5,7), ny+1)
        if (nx2 != nx+1) || (ny2 != ny+1)
            @warn "You are using a $base stochastic model with an image size of ($nx, $ny) which is not optimal for perf.\n" *
                  "I am changing the image size to ($(nx2-1), $(ny2-1)) which is optimal for $base"
            nx = nx2-1
            ny = ny2-1
        end
    end


    if polarized
        dp = Coherencies()
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
        dp = Visibilities()
        prep = TotalIntensity()
    end
    if polarized && !noleakage
        (isempty(array)) && throw(ArgumentError("If you are fitting polarized data with leakage, you must specify the array file"))
        obs = Pyehtim.load_uvfits_and_array(uvfile, array, polrep="circ")
    elseif polarized && noleakage
        obs = ehtim.obsdata.load_uvfits(uvfile, polrep="circ")
    else
        obs = ehtim.obsdata.load_uvfits(uvfile)
    end
    obs.add_scans()
    if scanavg
        obsavg = scan_average(obs.flag_uvdist(uv_min=uvmin))
    else
        obsavg = obs.flag_uvdist(uv_min=uvmin)
    end
    if space
        @warn "We are flagging space baselines as requested by the `--space` flag"
        obsavg = obsavg.flag_sites(["space"])
    end
    data = add_fractional_noise(extract_table(obsavg, dp), ferr)
    fix_nans_elevation!(data) # we need this because there are NaN's in the elevation
    x0 = μas2rad(x)
    y0 = μas2rad(y)
    hdr = ComradeBase.MinimalHeader(string(data.config.source),
        data.config.ra, data.config.dec,
        data.config.mjd, data[:baseline].Fr[1] # assume all frequencies are the same
    )
    if Threads.nthreads() > 1
        g = imagepixels(fovxrad, fovyrad, nx, ny, x0, y0; posang, executor=ThreadsEx(), header=hdr)
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
    skym = SkyModel(imgmod, skpr, g; algorithm=NonuniformFFTsAlg())

    nogains && polarized && throw(ArgumentError("--nogains flag with polarized imaging if not surported currently."))

    if !nogains
        if polarized && !noleakage
            intm = build_instrument_circular(; lgamp_sigma=lg, frcal)
        elseif polarized && noleakage
            intm = build_instrument_circularsimp(; lgamp_sigma=lg)
        elseif !gauto
            @info "Using the standard gain mode that assume IID gains with dispersion $lg"
            intm = build_instrument(; lgamp_sigma=lg)
        else
            @info "Using the automatic gain model that fits offsets and dispersion"
            intm = build_instrument_auto(; lgamp_sigma=lg)
        end
    else
        @info "You are assuming you have a perfect instrument"
        intm = Comrade.IdealInstrumentModel()
    end

    comrade_imager(
        data, outpath, skym, intm;
        nsample, nadapt,
        restart, benchmark,
        maxiters=maxiters, ntrials=ntrials, nimgs
    )
end
