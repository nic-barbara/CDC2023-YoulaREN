cd(@__DIR__)
using Pkg
Pkg.activate("./../..")

using BSON
using CairoMakie
using Revise
using StableRNGs

includet("./../../utils/load_utils.jl")
includet("../QubeServo/qube_servo_controller.jl")
includet("../QubeServo/qube_utils.jl")
includet("./utils.jl")

# Cost function
q1 = 5.0
q2 = 10.0
r1 = 0.01
R = reshape([r1], 1, 1)
vmax = 20

function cost_func(x::AbstractMatrix, u::AbstractMatrix)
    θ, α = x[1:1,:], x[2:2,:]
    Δu = clamp.(u, -vmax, vmax)

    zQz = 2*(q1*(1 .- cos.(θ)) + q2*(1 .+ cos.(α)))
    uRu = sum((R * Δu) .* Δu; dims=1)

    return mean(zQz + uRu)
end

# Some options
spath = "../../results/qube-robust/ecrit/"
fpath = "../../results/qube-experiment/"
fnames = [string(fpath,fname) for fname in readdir(fpath)]

# Load data
function get_qube_ecrit(fname; just_base=false, verbose=true)

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
    crit_val  = deg2rad(30)
    _func(ys) = mean(abs.(wrap_angles.(ys[end][2,:]) .- π))

    # Talk to user
    mtype = split(split(fname,"/")[end],"_")
    lab = mtype[1]
    v = mtype[end][1:2]
    println("Starting $lab $v:")

    # Test finding ϵcrit with grad_rescale_attack
    ϵcrit = find_special_ecrit(
        G, model, cost_func, x0, _func, crit_val; 
        ϵmax=1.0, thresh=1e-2, N=N, epochs=epochs,
        verbose=verbose
    )

    # Save the value
    save_name = string(spath, "ecrit_", split(fname,"/")[end])
    bson(save_name, Dict("ϵcrit" => ϵcrit))

    return ϵcrit
    
end

# Run it
get_qube_ecrit.(fnames)
get_qube_ecrit(fnames[1]; just_base=true)
