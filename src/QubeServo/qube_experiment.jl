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

    includet("./../../utils/load_utils.jl")
    includet("qube_servo_controller.jl")
    includet("qube_utils.jl")

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
    dt = 0.02
    max_steps = 100
    x0_lims = [π,π,0.5,0.5]

    # Construct the nonlinear model
    G = QubeServo2(;
        Δt = dt,
        max_steps = max_steps,
        x0_lims = x0_lims,
        rng = StableRNG(0)
    )

    # Get the linear model too
    Glin = linearised_qube2(;
        Δt = dt,
        max_steps = max_steps,
        x0_lims = x0_lims,
        rng = StableRNG(0),
        upwards=true
    )

    # Generate test data
    Batches = 100
    x0 = init_state(G, Batches; rng=StableRNG(seed))

    # Base controller
    q1 = 5.0
    q2 = 10.0
    q3 = 10
    q4 = q3

    r1 = 0.01
    vmax = G.v_max

    Q = diagm([q1, q2, q3, q4])
    R = reshape([r1], 1, 1)
    Cb = QubeCtrl(G, Glin.A, Glin.B, Q, R; k=600, ϵ=2dt)

    # Separate observer for some models
    obsv = TestHighGain(G; ϵ=4G.Δt)

    # Quadratic cost function (to get to upright)
    function cost_func(x::AbstractVector, u::AbstractVector)
        θ, α = x[1], x[2]
        Δu = clamp.(u, -vmax, vmax)
        return 2*(q1*(1 - cos(θ)) + q2*(1 + cos(α))) + Δu' * (R * Δu)
    end
    function cost_func(x::AbstractMatrix, u::AbstractMatrix)
        θ, α = x[1:1,:], x[2:2,:]
        Δu = clamp.(u, -vmax, vmax)

        zQz = 2*(q1*(1 .- cos.(θ)) + q2*(1 .+ cos.(α)))
        uRu = sum((R * Δu) .* Δu; dims=1)

        return mean(zQz + uRu)
    end

    # Test the base controller for comparison
    Jb = -rollout(G, Cb, cost_func, x0; rng=StableRNG(seed))
    
    # Where to save
    savedir = "../../results/qube-experiment/"

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
        "youla-ren-small",
        "youla-gren200-small",
        "youla-gren100-small",
        "youla-gren50-small",
    ]

    # Hyperparameters
    hp = Dict(
        "modeltype"      => model_types,
        "nx"             => 32,
        "nv"             => [28, 64, 64, 64, 64],
        "ϕfunc"          => Flux.relu,
        "div_factor"     => 8,

        "n_evals"        => 150000,
        "stepsize"       => 1e-2,
        "explore_mag"    => [2e-2, 5e-2, 5e-2, 5e-2, 5e-2],
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
