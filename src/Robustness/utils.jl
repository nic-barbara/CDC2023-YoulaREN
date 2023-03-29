using ColorSchemes
using Convex
using Flux, Zygote
using Mosek, MosekTools
using RobustNeuralNetworks
using Statistics

includet("../../utils/environments/load_environments.jl")

function LipSPD(m::WrapREN; verbose=false)

    # Define the variables
    P = Variable(m.nx, m.nx)
    λ = Variable(m.nv)
    γ = Variable()
    Λ = diagm(λ)

    # Q, S, R matrices
    Qinv = γ * I(m.ny)
    R = γ * I(m.nu)
    S = zeros(m.nu, m.ny)

    # Get all the matrices
    e = m.explicit
    W = 2Λ - Λ * e.D11 - e.D11' * Λ
    AB = [e.A'; e.B1'; e.B2']
    CD = [e.C2'; e.D21'; e.D22']

    # Construct the LMI matrices
    mat1 = [
        P           -e.C1' * Λ              e.C2' * S';
        -Λ*e.C1     W                       e.D21' * S' - Λ * e.D12;
        S * e.C2    S*e.D21 - e.D12' * Λ    R + S*e.D22 + e.D22' * S'
    ]
    mat2 = AB * P * AB'
    M = [mat1 - mat2 CD; CD' Qinv]

    # Add in the constraints. None on P for LBEN.
    # If this doesn't work on P for RREN, add P + (1e-6)*I(size(P)[1]) 
    constraints = Constraint[
        λ > 0,
        γ >= 0,
        M ⪰ 0
    ]
    (m.nx > 0) && push!(constraints, P ⪰ 0)

    # Solve the SDP
    problem = minimize(γ, constraints)
    solve!(problem, () -> Mosek.Optimizer(), silent_solver = !verbose)

    return problem.optval
end

function estimate_lipschitz_lower(
    model; 
    batches=10, 
    maxIter=400, 
    step_size=0.01, 
    clip_at=0.01, 
    init_var=0.001, 
    verbose=false,
    Tmax=100,
    nu=nothing
)

    # How many inputs?
    nu = (nu === nothing) ? model.nu : nu

    # Initial states and inputs
    x0 = init_state(model, batches)
    u1 = init_var * randn(nu, batches, Tmax)
    u2 = u1 .+ 1e-4 * randn(nu, batches, Tmax)

    # Set up optimisation params
    θ = Flux.Params([u1, u2, x0])

    # Flux optimiser
    opt = Flux.Optimiser(ClipValue(clip_at),
                        Flux.Optimise.ADAM(step_size),
                        Flux.Optimise.ExpDecay(1.0, 0.1, 200, 0.001)
                        )

    # Loss function
    function Lip()
        out = 0
        x1, x2 = x0, x0
        for t in 1:Tmax
            x1, y1 = model(x1, u1[:,:,t])
            x2, y2 = model(x2, u2[:,:,t])

            out += sum((y1 - y2).^2)
        end
        return sqrt(out) / norm(u1 - u2)
    end

    # Use gradient descent to estimate the Lipschitz bound
    lips = []
    for iter in 1:maxIter
        L, back = Zygote.pullback(() -> -Lip(), θ)
        ∇L = back(one(L))
        Flux.update!(opt, θ, ∇L)
        append!(lips, L)
        if (iter % 10 == 0) && verbose
            println("Iter: ", iter, "\t L: ", -L, "\t η: ", opt[3].eta)
        end
    end

    return maximum(-lips)
end

function estimate_upper_lower(model::WrapREN; nu=nothing, verbose=false)
    γmax = LipSPD(model; verbose=verbose)
    verbose && println("Upper bound: $γmax")
    γmin = estimate_lipschitz_lower(model; verbose=verbose)
    verbose && println("Lower bound: $γmin")
    return γmax, γmin
end

function estimate_upper_lower(model; nu=nothing, verbose=false)
    γmin = estimate_lipschitz_lower(model; nu=nu, verbose=verbose)
    verbose && println("Lower bound: $γmin")
    γmax = NaN
    return γmax, γmin
end

# Loss function for attacker
function loss(
    G, policy, costfunc, 
    xt, ξt, yt, ut; N=5
)

    cost = 0
    for t in 1:N
        ξt, ut = policy(ξt, yt, ut)
        cost += costfunc(xt, ut)
        xt = G(xt, ut)
        yt = measure(G, xt, ut)
    end

    return cost
end

# Attack a model
function attack_model(
    attack,
    G, policy, costfunc, 
    xt, ξt, yt, ut; 
    ϵ=0.1, N=5, epochs=1,
    rng=Random.GLOBAL_RNG
)

    for _ in epochs
        J = gradient(
            () -> loss(G, policy, costfunc, xt, ξt, yt, ut; N=N),
            Flux.params(yt)
        )

        yt .+= attack(J[yt], ϵ/epochs)
    end

    return yt
end

grad_rescale_attack(J, ϵ) = J .* (ϵ ./ (sqrt.(sum(J.^2,dims=1)) .+ 1e-12))

# Run a rollout
function attacked_rollout(
    G, policy, costfunc, x0, attack=fgsm_attack; 
    ϵ = 0.1, N = 5, epochs=1,
    verbose=false,
    rng=Random.GLOBAL_RNG
)

    cost = 0.0
    ξt = init_state(policy, size(x0,2); rng=rng)
    ut = init_ctrl(G, size(x0,2))
    xt = x0

    for t in 1:G.max_steps

        (verbose && t%10 == 0) && println("t: ",t, " cost: ", cost)

        # Get perturbed measurement
        yt = measure(G, xt, ut, t; rng=rng, noisy=true)
        yt = attack_model(
            attack, G, policy, costfunc, 
            xt, ξt, yt, ut; 
            ϵ=ϵ, N=N, epochs=epochs, rng=rng
        )

        # Controls, cost, and states
        ξt, ut = policy(ξt, yt, ut, t; rng=rng)
        cost += costfunc(xt, ut)
        xt = G(xt, ut, t; rng=rng, noisy=true)
    end

    return cost

end

function check_robustness(
    savepath, fname, cost_func, ϵs = [0.0]; 
    N = 5, epochs = 1, attack = grad_recale_attack, 
    thresh=1e-2, ϵmax=0.5,
    verbose=false, seed = 0,
    just_base = false,
    crit_cost = nothing
)

    # Load a model
    data  = BSON.load(fname)
    G     = data["Plant"]
    model = data["Model"]
    x0    = data["x0"]
    Q     = model.Q
    nu    = Q.nu

    # Should it just be the base controller?
    if just_base
        Q     = model.C
        model = model.C
        fname = "base-ctrl.bson"
    end

    # Remove random noise
    G.σv *= 0
    G.σw *= 0

    # Get estimates of lipschitz bound
    rnd(x) = round(x; digits=2)
    γmax, γmin = estimate_upper_lower(Q; nu=nu)
    verbose && println("Lipschitz bound estimates: max $(rnd(γmax)), min $(rnd(γmin))")

    # Unperturbed cost
    Jb = -rollout(G, model, cost_func, x0; rng=StableRNG(seed))
    verbose && println("Baseline cost: ", round(Jb; sigdigits=5))

    # Loop attack sizes in parallel
    costs = zeros(size(ϵs))
    Threads.@threads for k in eachindex(ϵs)
        costs[k] = attacked_rollout(
            G, model, cost_func, x0, attack; 
            ϵ = ϵs[k], N = N, epochs = epochs,
            rng = StableRNG(seed)
        )
        verbose && println(
            "Cost increase for ϵ = $(ϵs[k]): ", 
            rnd(costs[k]/Jb * 100),"%"
        )
    end

    # Estimate critical attack size separately
    ϵcrit = NaN

    # Save results
    save_robustness(savepath, fname, γmax, γmin, ϵs, costs, ϵcrit)

    return nothing

end

# Save attack information
function save_robustness(savepath, fname, γmax, γmin, ϵs, costs, ϵcrit=nothing)

    save_name = string(savepath, "rob_", split(fname,"/")[end])
    bson(
        save_name, 
        Dict(
            "γmax" => γmax,
            "γmin" => γmin,
            "ϵs" => ϵs,
            "costs" => costs,
            "ϵcrit" => ϵcrit,
        )
    )

    return nothing
end

get_stats(x::AbstractVector) = mean(x), minimum(x), maximum(x)

function get_stats(x::Vector{Vector{Float64}})
    μ = mean(x)

    y = zeros(length(x), length(μ))
    for k in eachindex(x)
        y[k,:] = x[k]
    end
    ymin = vec(minimum(y;dims=1))
    ymax = vec(maximum(y;dims=1))

    return μ, ymin, ymax
end

# Extract filename without "-vx.bson" attached
get_name(fname) = fname[1:end-8]

# Order files by non-lipschitz, then increasing Lipschitz
function order_files(fnames; ascending=false)

    # Store the non-Lipschitz files
    fnames1 = []
    indx = occursin.("gren", fnames)
    append!(fnames1, fnames[.!indx])
    n_nonlip = length(unique(get_name.(fnames1)))

    # Get Lipschitz bounds of the rest
    _get_lip(fname) = split(fname, "-")[2][5:end]
    lips = parse.(Int, _get_lip.(fnames[indx]))
    i_sort = sortperm(lips, rev = !ascending)

    # Order the Lipschitz files by Lipschitz bound
    append!(fnames1, fnames[indx][i_sort])
    return fnames1, n_nonlip
end

# Assign different plot markers for different model types
function get_marker(m)
    if occursin("gren", m)
        return :rect
    elseif occursin("ren", m)
        return :utriangle
    elseif occursin("lstm", m)
        return :circle
    else
        return :diamond
    end
end

# Assign different colours
function get_colour(m, k, n_nonlip; fudge_colours=false)
    if occursin("gren", m)
        return ColorSchemes.Greys_9[k - n_nonlip + 4]
    else
        fudge_colours && (k = (k+1) % 3 + 1)
        return ColorSchemes.Dark2_3[k]
    end
end

# Assign different line styles
function get_lstyle(m)
    if occursin("gren", m)
        return :solid
    elseif occursin("ren", m)
        return :dashdot
    elseif occursin("lstm", m)
        return :dash
    else
        return :dashdotdot
    end
end

# Assign different labels
function get_label(m)
    if occursin("gren",m)
        γ = split(m,"-")[2][5:end]
        return "Youla-γREN (γ = $γ)"
    elseif occursin("ren", m)
        return "Youla-REN (γ = ∞)"
    elseif occursin("lstm",m)
        return "Feedback-LSTM"
    else
        return "Base-Control"
    end
end


# ---------------------------------------------------
# 
#       Critical Attacks for the Paper
#
# ---------------------------------------------------

# Version of attacked rollout logs data
function attacked_rollout(
    G, policy, costfunc, x0, log_results::Bool;
    attack = grad_rescale_attack, 
    ϵ = 0.1, N = 5, epochs=1, verbose=false,
    rng=Random.GLOBAL_RNG
)

    # Set up data logging
    (x0 === nothing) && error("Must enter x0")
    batches = size(x0,2)
    x = fill(zeros(G.nx, batches), G.max_steps+1)
    u = fill(zeros(G.nu, batches), G.max_steps+1)
    y = fill(zeros(G.ny, batches), G.max_steps)

    # Initialise
    cost = 0.0
    ξt = init_state(policy, size(x0,2); rng=rng)
    u[1] = init_ctrl(G, size(x0,2))
    x[1] = x0

    for t in 1:G.max_steps

        (verbose && t%10 == 0) && println("t: ",t, " cost: ", cost)

        # Get perturbed measurement
        y[t] = measure(G, x[t], u[t], t; rng=rng, noisy=true)
        y[t] = attack_model(
            attack, G, policy, costfunc, 
            x[t], ξt, y[t], u[t]; 
            ϵ=ϵ, N=N, epochs=epochs, rng=rng
        )

        # Controls, cost, and states
        ξt, u[t+1] = policy(ξt, y[t], u[t], t; rng=rng)
        cost += costfunc(x[t], u[t+1])
        x[t+1] = G(x[t], u[t+1], t; rng=rng, noisy=true)
    end

    # Remove end points
    x = x[1:end-1]
    u = u[2:end]

    return cost, x, u, y

end

# Find ϵcrit at which func(ys) > crit_val
function find_special_ecrit(
    G, policy, costfunc, x0, func, crit_val; 
    N = 10, epochs = 10,
    ϵmax=0.5, maxiter=20, thresh=1e-2,
    verbose=false, rng=StableRNG(0)
)

    # Get average deviation from target
    function attacked_cost(ϵ) 
        _, _, _, ys = attacked_rollout(
            G, policy, costfunc, x0, true; 
            ϵ = ϵ, N = N, epochs = epochs, rng = rng
        )
        return func(ys)
    end

    # Initialise two starting guesses and a baseline
    ϵs = reshape([0.0, ϵmax],2,1)
    costs = attacked_cost.(ϵs)
    cost0 = crit_val

    # Bisection search
    iter = 1
    costs .-= cost0
    _rn(x) = round(x; digits=5)
    while (abs(ϵs[1] - ϵs[2]) > thresh) && (iter <= maxiter)

        verbose && println(
            "ϵ1: $(_rn(ϵs[1])), cost1: $(_rn(costs[1])), \tϵ2: $(_rn(ϵs[2])), cost2: $(_rn(costs[2]))"
        )

        ϵ_mid = [sum(ϵs)/2.0]
        cost_mid = attacked_cost.(ϵ_mid) .- cost0
        indx = sign.(cost_mid) .== sign.(costs)

        sum(indx) != 1 && (println("Bad value"); return ϵ_mid[1])

        ϵs[indx] = ϵ_mid
        costs[indx] = cost_mid
        iter += 1

    end
    
    return ϵs[2]

end
