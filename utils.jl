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

