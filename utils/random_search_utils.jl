using Flux
using LinearAlgebra
using Printf
using Revise
using StableRNGs

using ParameterSchedulers
using ParameterSchedulers: Stateful, next!

"""
Perform one rollout of a closed-loop system.

Introduces noise into the actual system state/measurements
"""
function rollout(G, policy, costfunc::Function, x0::AbstractMatrix; rng=Random.GLOBAL_RNG)
    cost = 0.0
    ξt = init_state(policy, size(x0,2); rng=rng)
    ut = init_ctrl(G, size(x0,2))
    xt = x0

    for t in 1:G.max_steps
        yt = measure(G, xt, ut, t; rng=rng, noisy=true)
        ξt, ut = policy(ξt, yt, ut, t; rng=rng)
        cost += costfunc(xt, ut)
        xt = G(xt, ut, t; rng=rng, noisy=true)
    end

    return -cost
end

# Version for plotting/investigation and debugging
function rollout(G, policy, costfunc::Function, returnall::Bool; x0=nothing, rng=Random.GLOBAL_RNG)
    ξt = init_state(policy; rng=rng)

    x = zeros(G.nx, G.max_steps+1)
    u = zeros(G.nu, G.max_steps+1)
    y = zeros(G.ny, G.max_steps)
    costs = zeros(G.max_steps)

    u[:,1] = init_ctrl(G)
    init_state(G; rng=rng)
    (x0===nothing) ? (x[:,1] = G.x0_lims) : (x[:,1] = x0)

    for t in 1:G.max_steps
        y[:,t] = measure(G, x[:,t], u[:,t], t; rng=rng, noisy=true)
        ξt, u[:,t+1] = policy(ξt, y[:,t], u[:,t], t; rng=rng)
        costs[t] = costfunc(x[:,t],u[:,t+1])
        x[:,t+1] = G(x[:,t], u[:,t+1],t; rng=rng, noisy=true)
    end
    x = x[:,1:end-1]
    u = u[:,2:end]

    return cumsum(costs), x, u, y
end


perturb_params!(p::AbstractVecOrMat, v::AbstractVecOrMat) = (p .+= v)

"""
Perturb a policy's parameters for random random_search
"""
function perturb_policy(model, σ::AbstractFloat; rng=Random.GLOBAL_RNG)

    π1, π2 = deepcopy(model), deepcopy(model)
    ps1, ps2 = Flux.params(π1), Flux.params(π2)

    ϵ = [randn(rng, size(p)...) for p in ps1]
    σϵ = σ.*ϵ
    
    perturb_params!.(ps1, σϵ)
    perturb_params!.(ps2, -σϵ)

    policy_reset!(π1)
    policy_reset!(π2)

    return π1, π2, ϵ
end

"""
Create dictionary equivalent to Zygote.Grads
"""
function grad_dict(gradvec::Vector, ps::Flux.Params)
    gs = Dict()
    for i in 1:length(ps)
        gs[ps[i]] = gradvec[i]
    end
    return gs
end

"""
Perform random search RL training on a controller (model)
given a system/plant (G) and cost function (costfunc).
"""
function random_search(model, G, costfunc;
    n_epochs        = 100,
    policy_batches  = 10,
    state_batches   = 10,
    explore_mag     = 0.01,
    stepsize        = 1e-3,
    max_grad_norm   = 10.0,
    step_decay_ep   = nothing,
    step_decay_mag  = 0.1,
    schedule_type   = :Step,
    verbose         = false,
    printfreq       = 10,
    rewardfreq      = 1,
    x0_test         = nothing,
    rng             = Random.GLOBAL_RNG #StableRNG(0)
)

    # Set up optimiser and learning rate scheduler
    (step_decay_ep === nothing) && (step_decay_ep = n_epochs+1)
    opt = Flux.Optimise.ADAM(stepsize)
    schedule = get_schedule(
        n_epochs, stepsize, step_decay_ep, step_decay_mag; 
        s_type=schedule_type
    )

    # Parameters and reward tracking
    ps = Flux.params(model)
    reward_nominal = zeros(ceil(Int,n_epochs/rewardfreq))
    (x0_test === nothing) && init_state(G, 100; rng=rng)

    # Copy env in a vector for multi-threading
    env_vec = [deepcopy(G) for _ in 1:policy_batches]

    for k in 1:n_epochs

        # Update the optimiser step size
        opt.eta = next!(schedule)

        # Track reward of unperturbed policy 
        if (rewardfreq == 1) || ((k % rewardfreq) == 1)

            kr = ceil(Int,k/rewardfreq)

            # Store at sampling rate (same RNG each time)
            r_nom = rollout(G, model, costfunc, x0_test; rng=StableRNG(0))
            reward_nominal[kr] = r_nom

            # Print at another rate
            if verbose && (kr % printfreq == 0) 
                @printf("Episode: %d, Reward: %.2f\n", k, r_nom)
            end

        end

        # Set up gradient estimate (same initial state for all)
        x0 = init_state(G, state_batches; rng=rng)
        reward_store = zeros(2*policy_batches)
        minibatch_store = fill(0 .* ps,policy_batches)

        # Multi-thread average gradient computations
        seed_vec = rand(rng, 1:100000, policy_batches)
        Threads.@threads for b in 1:policy_batches

            # Construct the two perturbed policies
            seed = seed_vec[b]
            π1, π2, ϵ = perturb_policy(model, explore_mag; rng=StableRNG(seed))

            # Run the system for each one with common x0/distrubance
            Gb = env_vec[b]
            r1 = rollout(Gb, π1, costfunc, x0; rng=StableRNG(seed))
            r2 = rollout(Gb, π2, costfunc, x0; rng=StableRNG(seed))

            # Store results
            minibatch_store[b] = (r1 - r2) .* ϵ
            reward_store[2b - 1] = r1
            reward_store[2b] = r2

        end
        minibatch = sum(minibatch_store)
        
        # Stop computation if wildly unstable
        σ_R = std(reward_store)
        if (σ_R > 1e10)
            println("Unstable gradients, breaking here.")
            break
        end

        # Estimate the gradient (negative for gradient descent on cost)
        my_grad = minibatch ./ (2 * explore_mag * policy_batches * σ_R)
        gs = grad_dict(-my_grad, ps)        

        # Update policy
        rescale_gradients!(gs, ps, max_grad_norm, L2norm)
        Flux.update!(opt, ps, gs)
        policy_reset!(model)

    end

    return reward_nominal, x0_test

end

"""
    get_schedule(n_epochs, α, decay_epoch, decay_mag; s_type=:Exp)

Get a scheduler to use for random search. Options are:
    1. :Step (from `α` to `α*decay_mag`) (Default)
    2. :Cos (cosine annealing, finishing on `α*decay_mag`)
Both schedulers reach their final value at `decay_epoch`.
"""
function get_schedule(n_epochs, α, decay_epoch, decay_mag; s_type=:Step)
    
    
    α_end = decay_mag*α
    
    n1 = decay_epoch
    n2 = n_epochs - decay_epoch
    
    if s_type == :Step
        s = Stateful(Sequence(α => n1, α_end => n2))
    elseif s_type == :Cos
        s1 = Cos(λ0 = α_end, λ1 = α, period = Int(n1/2))
        s2 = Exp(λ = α_end, γ = 1.0)
        s = Stateful(Sequence([s1,s2],[n1,n2]))
    else
        error("Unknown scheduler type $(s_type)")
    end

    return s

end

"""
Function for clipping/rescaling gradients based on specific norm
Based on code in ReinforcementLearningCore/src/extensions/Zygote.jl
"""
function rescale_gradients!(
    gs          :: AbstractDict, 
    ps          :: Flux.Params, 
    clip_norm   :: Number, 
    norm_func   :: Function
)
    norm∇L = norm_func(gs, ps)
    if clip_norm <= norm∇L
        clip_val = clip_norm / norm∇L
        for p in ps
            gs[p] .*= clip_val
        end
    end
    return norm∇L
end

L2norm(gs::AbstractDict, ps::Flux.Params) =
    sqrt(sum(mapreduce(x -> x^2, +, gs[p]) for p in ps))

L2norm(xs) = sqrt(sum(mapreduce(x -> x^2, +, x) for x in xs))
