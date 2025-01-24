using Pkg;
Pkg.activate(@__DIR__);
using Comonicon

using LinearAlgebra

include(joinpath(@__DIR__, "imaging_driver.jl"))
LinearAlgebra.BLAS.set_num_threads(1)
VLBISkyModels.FFTW.set_num_threads(1)
if Threads.nthreads() > 1
  VLBISkyModels.NFFT._use_threads[] = false
end


"""
Fits BHEX data using Comrade and ring prior for the image.

# Arguments
 - `uvfile::String`: the path to the uvfits file you want to fit

# Options

- `-o, --outpath`: the path to the output directory where images and other stats will be saved. Default is the current directory.
- `--fovx`: the field of view in microarcseconds. Default is 200 μas.
- `--fovy`: the field of view in microarcseconds. Default is fovx.
- `-p, --psize`: the pixel size in microarcseconds. Default is 1 μas.
- `-x, --x`: the x offset of image center in microarcseconds. Default is 0 μas.
- `-y, --y`: the y offset of image center in microarcseconds. Default is 0 μas.
- `--lftot`: the minimum flux density in Jy. Default is 0.2Jy.
- `--uftot`: the maximum flux density in Jy. Default is 2.5Jy.
- `-u, --uvmin`: the minimum uv distance in λ. Default is 0.2e9.
- `-n, --nimgs`: the number of image posterior samples to generate. Default is 200.
- `-a, --al`: the log-gain amplitude prior standard deviation. Default is 0.2.
- `--nsample`: the number of MCMC samples from the posterior. Default is 5_000.
- `--nadapt`: the number of MCMC samples to use for adaptation. Default is 2_500.
- `-f, --ferr`: the fractional error in the data. Default is 0.0.
- `--order`: the order of the Markov Random Field. Default is -1 which uses the Matern kernel.
- `--model`: The model to use for the prior image. Default is `:ring` other are `:isojet` and `:jet`.
             If `:isojet` is used, the there is a core with a isotropic extended emission. If `:jet` 
             is used, we fit the direction of the jet with a Gaussian. Note that `:jet` can be
             quite hard to fit.

# Flags

- `-r, ---restart`: Restart a previous checkpointed run assuming the checkpoint file is in the outpath.
- `-b, --benchmark`: Run a benchmarking test to see how long it takes to evaluate the logdensity and its gradient.
- `--scanavg`: Scan average the data prior to fitting. Note that is the data is merged multifrequency data, this will not work properly.
- `--space`: Flag space baselines. Namely this will flag any ground to space baselines.
"""
@main function main(uvfile::String; outpath::String="",
  fovx::Float64=200.0, fovy::Float64=fovx,
  psize::Float64=1.0,
  x::Float64=0.0, y::Float64=0.0,
  lftot::Float64=0.2, uftot::Float64=2.5,
  uvmin::Float64=0.2e9,
  nimgs::Int=200, al::Float64=0.2,
  model::String="ring",
  restart::Bool=false, benchmark::Bool=false, nsample::Int=5_000, nadapt::Int=2_500,
  scanavg::Bool=false,
  space::Bool=false,
  ferr::Float64=0.0,
  order::Int=-1
)

  fovxrad = μas2rad(fovx)
  fovyrad = μas2rad(fovy)
  nx = ceil(Int, fovx / psize)
  ny = ceil(Int, fovy / psize)

  outpath = isempty(outpath) ? first(splitext(uvfile)) : joinpath(outpath, first(splitext(basename(uvfile))))
  @info "Fitting the data: $uvfile"
  @info "Outputing to $outpath"
  @info "Field of view: ($fovx, $fovy) μas"
  @info "number of pixels: ($nx, $ny)"
  @info "Image center offset: ($x, $y) μas"
  @info "Adding $ferr fractional error to the data"

  if order < 0
    @info "Using Matern kernel for stochastic model"
    base = Matern()
  else
    @info "Using Markov Random Field of order $order for stochastic model"
    base = GMRF
  end



  obs = ehtim.obsdata.load_uvfits(uvfile)
  if scanavg
    obsavg = scan_average(obs.flag_uvdist(uv_min=uvmin))
  else
    obsavg = obs.flag_uvdist(uv_min=uvmin)
  end



  if !space
    data = add_fractional_noise(extract_table(obsavg, Visibilities()), ferr)
  else
    @warn "We are flagging space baselines as requested by the `--space` flag"
    data = add_fractional_noise(extract_table(obsavg.flag_sites(["space"]), Visibilities()), ferr)
  end

  x0 = μas2rad(x)
  y0 = μas2rad(y)
  hdr = ComradeBase.MinimalHeader(string(data.config.source), 
                                  data.config.ra, data.config.dec, 
                                  data.config.mjd, data[:baseline].Fr[1] # assume all frequencies are the same
                                )
  if Threads.nthreads() > 1
    g = imagepixels(fovxrad, fovyrad, nx, ny, x0, y0; executor=ThreadsEx(:dynamic), header=hdr)
  else
    g = imagepixels(fovxrad, fovyrad, nx, ny, x0, y0; header=hdr)
  end



  beam = beamsize(data)
  @info "Beam relative to pixel size: = $(beam/μas2rad(psize))"

  if model == "ring"
    imgmod = ImagingModel(TotalIntensity(), DblRingWBkgd(), g, Uniform(lftot, uftot); base, order)
    @info "Assuming the image is a ring"
  elseif model == "isojet"
    @info "Assuming the image is a isotropic jet structure"
    m = modify(Gaussian(), Stretch(beam/2))
    mimg = intensitymap(m, g)
    imgmod = ImagingModel(TotalIntensity(), MimgPlusBkgd(mimg ./ sum(mimg)), g, Uniform(lftot, uftot); base)
  elseif model == "jet"
    @info "Assuming the image is a anisotropic jet structure"
    m = modify(Gaussian(), Stretch(beam/2))
    mimg = intensitymap(m, g)
    imgmod = ImagingModel(TotalIntensity(), JetGauss(mimg./sum(mimg)), g, Uniform(lftot, uftot); base, order)
  else
    throw(ArgumentError("Unknown model: $model please pick from :ring, :isojet, :jet"))
  end



  skpr = skyprior(imgmod; beamsize=beam)
  skym = SkyModel(imgmod, skpr, g)
  intm = build_instrument(; lgamp_sigma=al)
  comrade_imager(
    data, outpath, skym, intm;
    nsample, nadapt,
    restart, benchmark,
    maxiters=15_000, ntrials=5, nimgs
  )
end


# Put imports here so that the CLI is snappier
