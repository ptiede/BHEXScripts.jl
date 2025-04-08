using Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "imaging_driver.jl"))

using Printf


function load_results(frame::Int, start=500)
    chains = map(3:9) do n
        in = @sprintf("runs/GroundStationTestStrictIncreaseGoodWeather/%dSites/frame%04d_merged_230.5-246.5_GHz_synthdata_ngEHTsim/", n, frame)
        load_samples(in)
    end

    posts = map(3:9) do n
        in = @sprintf("runs/GroundStationTestStrictIncreaseGoodWeather/%dSites/frame%04d_merged_230.5-246.5_GHz_synthdata_ngEHTsim_post.jls", n, frame)
        deserialize(in)[:post]
    end

    mss = map(posts, chains) do post, chain
        skymodel.(Ref(post), chain[start:end])
    end
    
    imgss = map(mss) do ms
        getproperty.(ms, :img)
    end    

    return chains, posts, imgss
end

function plot_imgs(posts, imgss; colorrange=(0.0, 4e-3))
    N = length(imgss)
    fig = Figure(;size=(100+160*N, 960))
    axs = [Axis(fig[i, j], xreversed=true, aspect=1) for i in 1:6, j in 1:N]
    hidedecorations!.(axs, grid=false)
    for i in 1:N
        axs[1, i].title = "$(i+1) Ground Sites"
    end

    for i in eachindex(posts)
        dt = datatable(posts[i].data[1])
        uv = dt.baseline
        CairoMakie.scatter!(axs[1, i],  uv.U,  uv.V, color=:black, markersize=3)
        CairoMakie.scatter!(axs[1, i], -uv.U, -uv.V, color=:black, markersize=3)
    end

    for i in eachindex(imgss)
        image!(axs[2, i], mean(imgss[i]); colormap=:afmhot, colorrange)
        image!(axs[3, i], mean(imgss[i])./std(imgss[i]), colormap=:viridis, colorrange=(0.0, 9.0))

    end
    Label(fig[2, 0], "Mean Image", rotation=π/2, tellheight=false, fontsize=18)
    Colorbar(fig[2, N+1]; colormap=:afmhot, colorrange=colorrange.*1e3, label="mJy/px²")

    Label(fig[3, 0], "SNR Map", rotation=π/2, tellheight=false, fontsize=18)
    Colorbar(fig[3, N+1], colormap=:viridis, colorrange=(0.0, 9.0), label="SNR")


    for j in eachindex(imgss), i in 4:6
        image!(axs[i, j], imgss[j][rand(1:length(imgss[j]))], colormap=:afmhot; colorrange)
    end
    Label(fig[4:6, 0], "Random Draws", rotation=π/2, tellheight=false, fontsize=18)
    hidespines!(axs[end])
    colgap!(fig.layout, 10,)
    rowgap!(fig.layout, 15,)
    colgap!(fig.layout, N+1, 0.1)

    fig
end



chains, posts, imgss = load_results(11, 2500);
fig = plot_imgs(posts, imgss, colorrange=(0.0, 5e-3))
save("Figures/frame0011.png", fig)

function single_baseline(dvis, bl)
    s = Set(bl)
    config = arrayconfig(dvis) |> datatable
    inds = findall(x->Set(x.sites)==s, config)


    return dvis[inds]
end

function flag_shortbaselines(dvis, uvmin)
    inds = findall(x->!(hypot(x.U, x.V) < uvmin), dvis[:baseline])
    bl = arrayconfig(dvis)[inds]
    coh = dvis[:measurement][inds]
    noise = dvis[:noise][inds]
    return EHTObservationTable{Comrade.datumtype(dvis)}(coh, noise, bl)
end

