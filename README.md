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

The current list of options are:

# Options

- `-o, --outpath`: the path to the output directory where images and other stats will be saved. Default is the current directory.
- `--fovx`: the field of view in microarcseconds. Default is 200 μas.
- `--fovy`: the field of view in microarcseconds. Default is fovx.
- `-p, --psize`: the pixel size in microarcseconds. Default is 1 μas.
- `-x, --x`: the x offset of image center in microarcseconds. Default is 0 μas.
- `-y, --y`: the y offset of image center in microarcseconds. Default is 0 μas.
- `-l, --lftot`: the minimum flux density in Jy. Default is 0.2Jy.
- `-u, --uftot`: the maximum flux density in Jy. Default is 2.5Jy.
- `-u, --uvmin`: the minimum uv distance in λ. Default is 0.2e9.
- `-n, --nimgs`: the number of image posterior samples to generate. Default is 200.
- `-a, --al`: the log-gain amplitude prior standard deviation. Default is 0.2.
- `-g, --gp`: the phase prior standard deviation. Default is π or uniform.
- `--nsample`: the number of MCMC samples from the posterior. Default is 5_000.
- `--nadapt`: the number of MCMC samples to use for adaptation. Default is 2_500.
- `-f, --ferr`: the fractional error in the data. Default is 0.0.

# Flags

- `-r, --restart`: Restart a previous checkpointed run assuming the checkpoint file is in the outpath.
- `-b, --benchmark`: Run a benchmarking test to see how long it takes to evaluate the logdensity and its gradient.
- `--scanavg`: Scan average the data prior to fitting. Note that is the data is merged multifrequency data, this will not work properly.
- `--jet`: Flag that we are considering jet imaging and not photon rings.
- `--space`: Flag space baselines. Namely if this options is used we will flag any ground to space baselines.
