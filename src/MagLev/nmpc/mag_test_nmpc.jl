cd(@__DIR__)
using Pkg
Pkg.activate("./../../..")

using BSON
using CairoMakie
using Ipopt
using JuMP
using Revise
using StableRNGs

includet("../../../utils/load_utils.jl")
includet("../mag_utils.jl")
includet("../output_fdback_backstep.jl")
includet("./mag_nmpc_controller.jl")

seed = 0
Random.seed!(seed)

# Cost is 29.457 with N = 30


# ----------------------------------------------------
#
#  Environment setup
#
# ----------------------------------------------------

# Environment parameters
dt = 0.005
max_steps = 100
x0_lims = [0.045, 0.5, 0.5]

# Construct the nonlinear model
G = MagLevEnv(;
    Δt = dt,
    max_steps = max_steps,
    x0_lims = x0_lims,
    rng = StableRNG(0)
)

# Generate test data
Batches = 100
x0 = init_state(G, Batches; rng=StableRNG(seed))

# Design a quadratic cost function 
yref = 0.05
iref = current_equilibrium(G,yref)
q1 = yref*0.5
r1 = 50.0
Q = diagm([1/q1^2, 0, 0])
R = diagm([1/r1^2])

function cost_func(x::AbstractVector, u::AbstractVector) 
    Δx = x .- [yref, 0, iref]
    Δu = clamp.(u, 0, 15) .- iref
    return ( Δx' * (Q * Δx) .+ Δu' * (R * Δu))
end

# The controller
N = 30
ctrl = MagLevCtrl(G, N; ϵ=dt)


# ----------------------------------------------------
#
#  Testing rollouts
#
# ----------------------------------------------------

function test_rollout(x0;just_cost=false)

    # Log states, controls
    xs = zeros(G.nx, G.max_steps+1)
    xo = zeros(ctrl.nx, G.max_steps+1)
    us = zeros(G.nu, G.max_steps)

    # Initial states
    xs[:,1] = x0[:,1]
    xo[:,1] = init_state(ctrl)

    # Simulate
    J = 0
    for t in 1:G.max_steps

        y = measure(G, xs[:,t]; noisy=true)
        xo[:,t+1], us[:,t] = ctrl(xo[:,t], y)
        J += cost_func(xs[:,t], us[:,t])
        xs[:,t+1] = G(xs[:,t], us[:,t]; noisy=true)

    end

    # Remove last state
    xs = xs[:,1:end-1]
    xo = xo[:,1:end-1]

    just_cost && (return J)
    return xs, us, J, xo

end

function sample_space(x0::AbstractMatrix; verbose=true)

    # Useful
    dt = G.Δt
    ts = dt:dt:G.max_steps*dt
    lwidth = 0.5
    nsims = size(x0,2)

    # Initialise 
    pos = zeros(nsims, G.max_steps)
    vel = zeros(nsims, G.max_steps)
    ctrl = zeros(nsims, G.max_steps)

    # Simulate
    J = 0
    for k in 1:nsims

        verbose && print("Starting sim $k of $nsims... ")

        ys, us, Jk, _ = test_rollout(x0[:,k])
        pos[k,:] = ys[1,:]
        vel[k,:] = ys[2,:]
        ctrl[k,:] = us[1,:]
        J += Jk

        println("Done!")
    end

    # Construct figure
    size_inches = (8, 4)
    size_pt = 100 .* size_inches
    f = Figure(resolution = size_pt, fontsize = 16)
    ga = f[1,1] = GridLayout()
    
    # Plot position
    ax1 = Axis(ga[1,1], xlabel = "Time (s)", ylabel = "Vertical height (m)")
    for k in axes(pos,1)
        lines!(ax1, ts, pos[k,:], linewidth=lwidth, color=:grey)
    end
    ylims!(ax1, 0.0, 0.1)
    ax1.yreversed = true

    # Plot control input
    ax3 = Axis(ga[1,2], xlabel = "Time (s)", ylabel = "Control (V)")
    for k in axes(pos,1)
        lines!(ax3, ts, ctrl[k,:], linewidth=lwidth, color=:grey)
    end

    display(f)

    return J/nsims

end

J = sample_space(x0)

# Print the cost
println("NMPC cost: ", round(J; sigdigits=5))
