mutable struct FCallback{F}
    counter::Int
    stride::Int
    const f::F
end
FCallback(stride, f) = FCallback(0, stride, f)
function (c::FCallback)(x, others)
    c.counter += 1
    if c.counter % c.stride == 0
        @info "On step $(c.counter) f = $(x.objective)"
        return false
    else
        return false
    end
end

function best_image(post, ntrials=20, maxiters=10_000, rng=rng)
    sols = map(1:ntrials) do i
        xopt0, sol0 = comrade_opt(post, Adam();
                           initial_params=prior_sample(rng, post), maxiters=maxiters÷2, g_tol=1e-1)
        @info "Preliminary image $i/$(ntrials) done: minimum: $(sol0.minimum)"

        xopt1, sol1 = comrade_opt(post, Adam();
                           initial_params=xopt0, maxiters=maxiters÷2, g_tol=1e-1)
        @info "Best image $i/$(ntrials) done: minimum: $(sol1.minimum)"
        return (sol0.minimum < sol1.minimum ? xopt0 : xopt1)
    end
    lmaps = sum.(logdensityof.(Ref(post), sols))
    inds = sortperm(filter(!isnan, lmaps), rev=true)
    return sols[inds], lmaps[inds]
end

function fix_nans_elevation!(data)
    el1 = data.config.datatable.elevation.:1
    el2 = data.config.datatable.elevation.:2
    for i in eachindex(el1, el2)
        isnan(el1[i]) && (el1[i] = 0.0)
        isnan(el2[i]) && (el2[i] = 0.0)
    end
end

function load_chain_and_post(cfile, itr=Colon())
    chain = load_samples(cfile, itr)
    post = deserialize(cfile*"_post.jls")[:post]
    return chain, post
end

function make_df(chain)
    df = DataFrame()
    csub = chain[rand(1:length(chain), 500)]
    df[!, :ellip_1] = csub.sky.τ1
    df[!, :ellip_0] = csub.sky.τ0
    df[!, :ellip_angle_1] = csub.sky.ξτ1
    df[!, :ellip_angle_0] = csub.sky.ξτ0
    df[!, :x1] = csub.sky.x1
    df[!, :y1] = csub.sky.y1
    df[!, :gamma_width] = csub.sky.γ
    df[!, :gamma_flux] = csub.sky.γf
    df[!, :r1] = csub.sky.r1
    df[!, :r0] = csub.sky.r0
    return df
end

function make_bhex_output(cfile, itr=Colon())
    chain, post = load_chain_and_post(cfile, itr)
    df = make_df(chain)
    CSV.write(cfile*"_bhex.csv", df)
    ms = skymodel.(Ref(post), chain[1:5:end])
    gpl = refinespatial(post.skymodel.grid.imgdomain, 3)
    imgs = intensitymap.(ms, Ref(gpl))
    save_fits(cfile*"_bhex.fits", mean(imgs))
    return df, imgs
end