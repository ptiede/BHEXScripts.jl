using Pkg;Pkg.activate(@__DIR__)
using Pyehtim
using CairoMakie
using Comrade
using AdvancedHMC
using Optimization
using OptimizationOptimisers
using Random
using Distributions
using VLBIImagePriors
using Enzyme
using CSV
using Serialization
using BenchmarkTools
using LinearAlgebra
include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "skymodels.jl"))
include(joinpath(@__DIR__, "skymodels2.jl"))
include(joinpath(@__DIR__, "instrumentmodels.jl"))
Enzyme.EnzymeRules.inactive_type(::Type{<:Comrade.EHTObservationTable}) = true



function comrade_imager(data, outbase, skym, intm; maxiters=15_000, ntrials=10,
              nsample = 10_000, nadapt = 5_000, rng = Random.default_rng(), restart=false,
              start = nothing,
              benchmark=true, nimgs=500)

    mkpath(dirname(outbase))
    imgout = joinpath(mkpath(joinpath(dirname(outbase), "images")), basename(outbase)*"_results")
    @info "Outputing to $outbase"


    post  = VLBIPosterior(skym, intm, add_fractional_noise(data, 0.0); admode=set_runtime_activity(Enzyme.Reverse))
    tpost = asflat(post)
    ndim = dimension(tpost)

    if benchmark

        @info "Forward Pass benchmark"
        x0 = randn(ndim)
        btt = @benchmark logdensityof($tpost, $x0)
        io = IOContext(stdout)
        show(io, MIME("text/plain"), btt)
        println()

        @info "Reverse Pass benchmark"
        btt = @benchmark Comrade.LogDensityProblems.logdensity_and_gradient($tpost, $x0)
        io = IOContext(stdout)
        show(io, MIME("text/plain"), btt)
        println()

    end

    out = outbase
    g = post.skymodel.grid.imgdomain


    if !restart && isnothing(start)

        # Start by essentially tempering the likelihood
        post0 = VLBIPosterior(skym, intm, add_fractional_noise(data, 0.05); admode=set_runtime_activity(Enzyme.Reverse)) 
        sols, ℓopt = best_image(post0, ntrials, maxiters, rng)

        # Now lets get a little closer to the truth
        post1 = VLBIPosterior(skym, intm, add_fractional_noise(data, 0.025); admode=set_runtime_activity(Enzyme.Reverse))
        xopt1, sol = comrade_opt(post1, Adam();
                           initial_params=sols[1], maxiters=maxiters÷2, g_tol=1e-1)
        
        r = residuals(post1, xopt1)
        fig = plotfields(r[1], :uvdist, :res)
        save(out*"_residuals_step1_map.png", fig)

        
        post2 = VLBIPosterior(skym, intm, add_fractional_noise(data, 0.01); admode=set_runtime_activity(Enzyme.Reverse))
        xopt2, sol = comrade_opt(post2, Adam();
                           initial_params=xopt1, maxiters=maxiters÷2, g_tol=1e-1)

        r = residuals(post2, xopt2)
        fig = plotfields(r[1], :uvdist, :res)
        save(out*"_residuals_step2_map.png", fig)
        xopt, sol = comrade_opt(post, Adam();
                            maxiters=maxiters, g_tol=1e-1,
                            initial_params=xopt2)

        img = intensitymap(skymodel(post, xopt), g)

        Comrade.save_fits(imgout*"_optimal.fits", img)
        p = imageviz(img)
        CairoMakie.save(imgout*"_optimal.png", p)


        r = residuals(post, xopt)
        fig = plotfields(r[1], :uvdist, :res)
        save(out*"_residuals_final_map.png", fig)

        if hasproperty(xopt, :instrument)
            k = keys(xopt.instrument)
            v = values(xopt.instrument)
            map(k, v) do ki, vi
                gtp = Comrade.caltable(vi)
                CSV.write(out*"_ctable_$ki.csv", gtp)
                fig = plotcaltable(gtp)
                CairoMakie.save(out*"_ctable_$ki.png", fig)
            end
        end
    elseif restart && isnothing(start)
        xopt = deserialize(out*"_post.jls")[:xopt]
    else
        @info "Using passed initial location"
        xopt = start
    end

    serialize(out*"_post.jls", Dict(
        :xopt=>xopt,
        :post=>post))




    trace = sample(rng, post, NUTS(0.8), nsample; 
                   saveto=DiskStore(mkpath(out), 25), 
                   initial_params = xopt,
                   n_adapts=nadapt,
                   restart
                   )
    if restart
        chain = load_samples(out, nadapt+1:10:nsample)
    else
        chain = load_samples(trace, nadapt+1:10:nsample)
    end

    ss = sample(chain, 10)
    p = residuals(post, ss[begin])
    fig, ax = baselineplot(p[1], :uvdist, :res)
    for s in ss[2:end]
        baselineplot!(ax, residuals(post, s)[1], :uvdist, :res)
    end
    ax.title = "χ² = $(mean(getindex.(chi2.(Ref(post), ss),1))/(2*length(post.data[1])))"
    CairoMakie.save(out*"_residuals.png", fig)
    # Finally let's construct some representative image reconstructions.

    samples = skymodel.(Ref(post), sample(chain, nimgs))
    imgs = intensitymap.(samples, Ref(g));

    for (i, img) in enumerate(imgs)
        Comrade.save_fits(imgout*"_draw$i.fits", img)
        p = imageviz(img)
        CairoMakie.save(imgout*"_draw$i.png", p)
    end
    mimg = mean(imgs)
    simg = std(imgs)
    Comrade.save_fits(imgout*"_mean_image.fits", mimg)
    Comrade.save_fits(imgout*"_std_image.fits", simg)
end