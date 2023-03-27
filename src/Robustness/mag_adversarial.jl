cd(@__DIR__)
using Pkg
Pkg.activate("./../..")

using BSON
using CairoMakie
using Revise
using StableRNGs

includet("./../../utils/load_utils.jl")
includet("../MagLev/output_fdback_backstep.jl")
includet("../MagLev/mag_utils.jl")
includet("./utils.jl")

seed = 0
Random.seed!(seed)

# Base controller
yref = 0.05
iref = 6.2642
q1 = yref*0.5
r1 = 50.0
Q = diagm([1/q1^2, 0, 0])
R = diagm([1/r1^2])

# Cost function
function cost_func(x::AbstractMatrix, u::AbstractMatrix)
    Δx = x .- [yref, 0, iref]
    Δu = clamp.(u, 0, 15) .- iref

    xQx = sum((Q * Δx) .* Δx; dims=1)
    uRu = sum((R * Δu) .* Δu; dims=1)

    return mean(xQx + uRu)
end

# Some options
fpath = "../../results/mag-experiment/"
fnames = [string(fpath,fname) for fname in readdir(fpath)]

# Set up attacks and saving
N = 10
epochs = 10
verbose = true
ϵs = range(0.0, stop=0.05, length=60)
attack = grad_rescale_attack
savepath = "../../results/mag-robust/experiment/"

# Cost at which to find attack size to higher precision (for plotting)
Jb = 45.66
crit_cost = 1.8*Jb

# Check robustness for each model
_f(fname; base=false) = check_robustness(
    savepath, fname, cost_func, ϵs; epochs = epochs,
    N = N, attack = attack, verbose = verbose, seed = seed,
    thresh = 2e-4, ϵmax = 0.1, just_base = base, crit_cost = crit_cost
)

_f.(fnames)
_f(fnames[1]; base=true)
