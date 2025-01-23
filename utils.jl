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


function add_fractional_noise(dvis, ferr)
    dvis2 = deepcopy(dvis)
    map!(dvis2[:noise], dvis2[:noise], dvis2[:measurement]) do e, m
        fe =  sqrt.(e.^2 .+ ferr.^2*abs2(tr(m))/2)
        return fe
    end
    return dvis2
end

function flag_shortbaselines(dvis, uvmin)
    inds = findall(x->!(hypot(x.U, x.V) < uvmin), dvis[:baseline])
    bl = arrayconfig(dvis)[inds]
    coh = dvis[:measurement][inds]
    noise = dvis[:noise][inds]
    return EHTObservationTable{Comrade.datumtype(dvis)}(coh, noise, bl)
end

function flag_baselines(dvis, baselines...)
    bls = map(Set, baselines)
    inds = findall(x->!any(y->(Set(x.baseline) == y), bls), dvis.data)

    c = arrayconfig(dvis)
    config = Comrade.EHTArrayConfiguration(c.bandwidth, c.tarr, c.scans, c.data[inds])
    data = dvis[inds]

    return Comrade.EHTObservation(;
                data, mjd = dvis.mjd,
                ra = dvis.ra, dec = dvis.dec,
                config,
                bandwidth = dvis.bandwidth,
                source = dvis.source
            )

end


function select_time(dvis, tlower, tupper)
    inds = findall(x->(tlower<=x.T<tupper), dvis.data)

    c = arrayconfig(dvis)
    config = Comrade.EHTArrayConfiguration(c.bandwidth, c.tarr, c.scans, c.data[inds])
    data = dvis[inds]

    return Comrade.EHTObservation(;
                data, mjd = dvis.mjd,
                ra = dvis.ra, dec = dvis.dec,
                config,
                bandwidth = dvis.bandwidth,
                source = dvis.source
            )

end

function select_baselines(dvis, bls::NTuple{2, Symbol}...)
    s = Set.(bls)
    config = arrayconfig(dvis) |> datatable
    inds = findall(x->Set(x.sites)∈s, config)
    return dtbl[inds]
end

function single_baseline(dvis, bl)
    s = Set(bl)
    config = arrayconfig(dvis) |> datatable
    inds = findall(x->Set(x.sites)==s, config)


    return dvis[inds]
end

function flag_site(dvis::Comrade.EHTObservationTable{T}, site) where {T}
    config = arrayconfig(dvis) |> datatable
    inds = findall(x->!(site in x.sites), config)
    m = measurement(dvis)[inds]
    s = noise(dvis)[inds]
    conf = arrayconfig(dvis)[inds]
    return Comrade.EHTObservationTable{T}(m, s, conf)
end