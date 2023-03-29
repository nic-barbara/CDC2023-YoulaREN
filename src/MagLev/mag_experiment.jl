using Distributed
# Open Julia with: julia -t 4 -p 4 for 4 workers, each with 4 threads

# Activate package for all workers
@everywhere begin 
    cd(@__DIR__)
    using Pkg
    Pkg.activate("./../..")
end

# Load packages
@everywhere begin
    using Revise
    using StableRNGs

    includet("../../utils/load_utils.jl")
    includet("./output_fdback_backstep.jl")
    includet("./mag_utils.jl")

    seed = 0
    Random.seed!(seed)
end


# ----------------------------------------------------
#
# Setup: environment, cost, backup control
#
# ----------------------------------------------------

@everywhere begin
    
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

    # Base controller
    yref = 0.05
    Cb = MagLevCtrl(G; yref=yref, ϵ=dt)
    obsv = MagLevObsv(G; ϵ=3dt)

    # Design a quadratic cost function 
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
    function cost_func(x::AbstractMatrix, u::AbstractMatrix)
        Δx = x .- [yref, 0, iref]
        Δu = clamp.(u, 0, 15) .- iref

        xQx = sum((Q * Δx) .* Δx; dims=1)
        uRu = sum((R * Δu) .* Δu; dims=1)

        return mean(xQx + uRu)
    end

    # Test the base controller for comparison
    Jb = -rollout(G, Cb, cost_func, x0; rng=StableRNG(seed))
    
    # Where to save
    savedir = "../../results/mag-experiment/"

end

# Print the base cost
println("Baseline cost: ", round(Jb; sigdigits=5))


# ----------------------------------------------------
#
# Set up experiments
#
# ----------------------------------------------------

@everywhere begin 

    # Models to tune
    model_types = [
        "feedback-lstm",
        "youla-ren-obsv-small",
        "youla-gren2000-obsv-small",
        "youla-gren1000-obsv-small",
        "youla-gren500-obsv-small",
        "youla-gren250-obsv-small",
    ]

    # Hyperparameters
    hp = Dict(
        "modeltype"      => model_types,
        "nx"             => 32,
        "nv"             => [28, 64, 64, 64, 64, 64],
        "ϕfunc"          => Flux.relu,
        "div_factor"     => 8,

        "n_evals"        => 150000,
        "stepsize"       => [1e-2, 5e-3, 5e-3, 5e-3, 5e-3, 5e-3],
        "explore_mag"    => [1e-2, 1e-2, 1e-2, 5e-2, 5e-2, 5e-2],
        "state_batches"  => 50,
        "policy_batches" => 16,
        "grad_clip"      => 10,

        "step_decay"     => 0.7,
        "n_experiments"  => 6,
        "reward_freq"    => 20,
    )
    hp_set = get_hp_dicts(hp)

    # Build a list of experiment options
    experiment_list = Vector{Any}([])
    for i in eachindex(hp_set)
        append!(
            experiment_list, 
            [ExperimentParams(
                deepcopy(G),
                x0,
                cost_func,
                Cb,
                hp_set[i],
                (Jb),
                savedir;
                model_init = obsv,
                version_num = e
            ) for e in 1:hp_set[i]["n_experiments"]]
        )
    end
end


# ----------------------------------------------------
#
# Run experiments
#
# ----------------------------------------------------

println("Starting $(length(experiment_list)) experiments...")
pmap(run_experiment_manual_loop, experiment_list)
println("All done.")
