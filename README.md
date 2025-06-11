# BHEX Comrade Imaging Script

This script is for generic imaging of BHEX synthetic data. To use this we assume that you
have Julia 1.10 installed. If you do not we recommend installing Julia with `juliaup`
https://github.com/JuliaLang/juliaup. If you have juliaup install to use Julia 1.10 run
the following commands

```
juliaup add lts
juliaup default lts
```

**Note that the script will not work with any Julia version aside from 1.10.**


The main script to run is `main.jl`. To use it first call `setup.jl`:

```
julia setup.jl
```

which will install all the required dependencies and precompile the scripts. **If you make any changes to the script,
please run the `setup.jl` script again.**

To image a dataset you can run the command
```
julia main.jl /path/uvfits/you/want/to/fit -o /path/to/folder/for/output ...
```

Fits BHEX data using Comrade and ring prior for the image.

## Arguments

- `uvfile::String`: the path to the uvfits file you want to fit

## Options

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
- `--order`: the order of the Markov Random Field. Default is -1 which uses the Matern kernel.
- `--model`: The model to use for the prior image. Default is `ring` others are `ringnojet`, `isojet`, `jet`, and `flat`. The options mean:
  - `ring` is a ring with a constant background.
  - `ringnojet` is a ring with no constant background. 
  - `isojet` is a core with a constant floor that is fit. 
  - `jet` is fits the the jet with a asymmetric Gaussian. Note that `:jet` can be quite hard to fit. 
  - `flat` is a flat image with no structure.
- `--maxiters`: the maximum number of iterations for the optimizer. Default is 15_000.
- `--ntrials`: the number of trials to run the optimizer. Default is 10.
- `--polrep`: The polarization representation. The default of PolExp which uses a matrix exponential representation.
- `--refsite`: The reference site for EVPA calibration. Default is `ALMA`.

## Flags

- `-r, ---restart`: Restart a previous checkpointed run assuming the checkpoint file is in the outpath.
- `-b, --benchmark`: Run a benchmarking test to see how long it takes to evaluate the logdensity and its gradient.
- `--scanavg`: Scan average the data prior to fitting. Note that if the data is merged multifrequency data, this will not work properly.
- `--space`: Flag space baselines. Namely this will flag any ground to space baselines.
- `--polarized`: Fit the polarized data. This requires the `--array` flag to be set.
- `--frcal`: Flag that the data has been FR-cal'd (not the default in ngehtsim)
- `--noleakage`: Assumes that the data doesn't have leakage.
- `--nogains`: Assumes that the instrument is perfect and does not have any gains.  
