using Pkg; Pkg.activate(@__DIR__)
using Pyehtim
using Glob

function merge_data(filef1, freq1, freq2="246.5")
    occursin(freq1, filef1) || throw(error("Frequency $freq1 not found in $filef1"))
    obs1 = ehtim.obsdata.load_uvfits(filef1)
    filef2 = replace(filef1, freq1=>freq2)
    obs2 = ehtim.obsdata.load_uvfits(filef2)
    obs2.data["time"] = obs2.data["time"] + 0.1
    obs = ehtim.merge_obs((obs1, obs2), force_merge=true)
    obs.tarr = obs1.tarr
    @info "Output to $(replace(filef1, freq1=>"merged_$freq1-$freq2"))"
    obs.save_uvfits(replace(filef1, freq1=>"merged_$freq1-$freq2"))
end

# n = 3:9
# for i in n
files = readdir(Glob.GlobMatch("data/Simulation70/singlenight_obs/*230.5*ngEHTsim.uvfits"))
@info files
merge_data.(files, "230.5")
# end