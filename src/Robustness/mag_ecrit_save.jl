cd(@__DIR__)
using Pkg
Pkg.activate("./../..")

using BSON
using CairoMakie
using Random
using Revise
using StableRNGs

includet("./../../utils/load_utils.jl")
includet("../MagLev/output_fdback_backstep.jl")
includet("../MagLev/mag_utils.jl")
includet("./utils.jl")

Random.seed!(0)

# Cost function
yref = 0.05
iref = 6.2642
q1 = yref*0.5

r1 = 50.0
Q = diagm([1/q1^2, 0, 0])
R = diagm([1/r1^2])

function cost_func(x::AbstractMatrix, u::AbstractMatrix)
    Δx = x .- [yref, 0, iref]
    Δu = clamp.(u, 0, 15) .- iref

    xQx = sum((Q * Δx) .* Δx; dims=1)
    uRu = sum((R * Δu) .* Δu; dims=1)

    return mean(xQx + uRu)
end

# Some options
spath = "../../results/mag-robust/ecrit/"
fpath = "../../results/mag-experiment/"
fnames = [string(fpath,fname) for fname in readdir(fpath)]

# Function to repeat
function get_mag_ecrit(fname; just_base=false, verbose=true)

    # Load data
    data  = BSON.load(fname)
    G     = data["Plant"]
    model = data["Model"]
    x0    = data["x0"]
    
    if just_base 
        model = model.C
        fname = "base-ctrl.bson"
    end

    # Remove random noise
    G.σv *= 0
    G.σw *= 0

    # Attack options
    N         = 10
    epochs    = 10
    crit_val  = 0.01
    _func(ys) = mean(abs.(ys[end][1,:] .- 0.05))

    # Talk to user
    mtype = split(split(fname,"/")[end],"_")
    lab = mtype[1]
    v = mtype[end][1:2]
    println("Starting $lab $v:")

    # Test finding ϵcrit with grad_rescale_attack
    ϵcrit = find_special_ecrit(
        G, model, cost_func, x0, _func, crit_val; 
        ϵmax=0.05, thresh=1e-3, N=N, epochs=epochs,
        verbose=verbose
    )

    # Save the value
    save_name = string(spath, "ecrit_", split(fname,"/")[end])
    bson(save_name, Dict("ϵcrit" => ϵcrit))

    return ϵcrit
end

# Run it
get_mag_ecrit.(fnames)
get_mag_ecrit(fnames[1]; just_base=true)
