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

seed = 0
Random.seed!(seed)

# Base controller
q1 = 5.0
q2 = 10.0
r1 = 0.01
R = reshape([r1], 1, 1)
vmax = 20

# Cost function
function cost_func(x::AbstractMatrix, u::AbstractMatrix)
    θ, α = x[1:1,:], x[2:2,:]
    Δu = clamp.(u, -vmax, vmax)

    zQz = 2*(q1*(1 .- cos.(θ)) + q2*(1 .+ cos.(α)))
    uRu = sum((R * Δu) .* Δu; dims=1)

    return mean(zQz + uRu)
end

# Some options
fpath = "../../results/qube-experiment/"
fnames = [string(fpath,fname) for fname in readdir(fpath)]

# Set up attacks and saving
N = 10
epochs = 10
verbose = true
ϵs = range(0.0, stop=0.8, length=60)
attack = grad_rescale_attack
savepath = "../../results/qube-robust/experiment/"

# Cost at which to find attack size to higher precision (for plotting)
Jb = 1534
crit_cost = 1.25*Jb

# Check robustness for each model
_f(fname; base=false) = check_robustness(
    savepath, fname, cost_func, ϵs; epochs = epochs,
    N = N, attack = attack, verbose = verbose, seed = seed,
    thresh=2e-3, ϵmax = 1.0, just_base = base, crit_cost = crit_cost,
)

_f.(fnames)
_f(fnames[1]; base=true)
