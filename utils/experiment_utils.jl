using BSON
using CairoMakie
using Dates
using DocStringExtensions
using Printf
using Revise
using RobustNeuralNetworks
using StatsBase
using TypedTables

includet("./models/load_models.jl")
includet("./get_model.jl") # for the function get_model()

"""
    my_float2str(x)

Function converts a float to a string by removing the
decimal place and padding with zeros. Eg: 0.01 becomes `"01"`,
0.001 becomes `"001"`
"""
function my_float2str(x)
    n0 = floor(Int,log10(x))
    return lpad(Int(x*10^(-n0)),-n0,"0")
end

"""
$(TYPEDEF)

All the information required to run an experiment
"""
struct ExperimentParams
    env                         # Plant dynamics/environment 
    env_model                   # Model of plant dynamics for controller to see
    x0_test                     # Initial states for testing
    cost_func                   # Cost function
    backup                      # Backup controller
    hyperparams                 # Hyperparameter dictionary
    model_init                  # Initialiser for the model (only used for REN)
    base_costs                  # Base controller costs (handy to store)
    savedir                     # Directory to save results
    version_num                 # Version number (optional)
end

"""
Custom initialiser for `ExperimentParams`
"""
function ExperimentParams(
    env, x0_test, cost_func, backup,
    hyperparams, base_costs, savedir; 
    model_init=nothing, version_num=0
)
    ExperimentParams(
        env, deepcopy(env), x0_test, cost_func, 
        backup, hyperparams, model_init, base_costs, 
        savedir, version_num
    )
end

"""
    init_hp_dict()

Get a dictionary of default hyperparameters for
training feedback models with random search.
"""
function init_hyperparams()
    Dict(

        "modeltype" => "",

        # Model setup
        "γ" => NaN,
        "nx" => 32,
        "nv" => 64,
        "ϕfunc" => Flux.relu,
        "div_factor" => 1,
        "filtertype" => NoFilter,

        # Random search
        "n_evals" => 1000,
        "stepsize" => 1e-3,
        "grad_clip" => 10,
        "explore_mag" => 1e-3,
        "state_batches" => 10,
        "policy_batches" => 25,

        # Learning rate scheduling
        "step_decay" => 1.0,
        "schedule_type" => :Step,

        # Printing and logging, repeats
        "printfreq" => 1,
        "reward_freq" => 10,
        "n_experiments" => 1,
    )
end

"""

Construct a vector of hyperparameter dictionaries given a range
of options. Rules are as follows:

- For a given `"modeltype"`, any field can be a vector of hyperparams
- For a vector in the `"modeltype"` field, hyperparams can either be: 
a) a number (same for all models) 
b) a vector of the same length as the number of modeltypes
c) a vector of vectors, where each subvector is a range of params to try on that model

Every permutation of hyperparameters is used, so be careful not to go crazy!
This should allow plenty of flexibility for experiments and model tuning now.
"""
function get_hp_dicts(hp_in::Dict)

    hp_base = init_hyperparams()

    # Figure out which fields are vector-valued or not
    keys = []
    vals = []
    isvec = zeros(Bool, length(hp_in))

    i = 1
    for (key, value) in hp_in
        push!(keys, key)
        push!(vals, value)
        isvec[i] = typeof(value) <: AbstractVector
        i += 1
    end

    # Fill in the non-vector fields
    for k in keys[.!isvec]
        hp_base[k] = hp_in[k]
    end

    # Return if that's it (one model, one hp set)
    if !any(isvec) 
        set_gamma!(hp_base)
        return [hp_base]
    end

    # Loop the different model configurations
    hp_set = []
    models = hp_in["modeltype"]
    !(typeof(models) <: AbstractVector) && (models = [models])

    # Build vectors of values/keys that have multiple options
    valsvec = vals[isvec]
    keysvec = keys[isvec]
    valsvec = valsvec[keysvec .!= "modeltype"]
    keysvec = keysvec[keysvec .!= "modeltype"]

    # Loop the models
    for i in eachindex(models)

        # Hyperparameters for this model
        hp_model = deepcopy(hp_base)
        hp_model["modeltype"] = models[i]
        set_gamma!(hp_model)

        # Vector of hyperparam values for this model
        valsvec_i = [valsvec[j][i] for j in eachindex(valsvec)]

        # Loop through all combinations and create separate HP dicts
        for v in Iterators.product(valsvec_i...)

            hp = deepcopy(hp_model)
            for j in eachindex(v)
                hp[keysvec[j]] = v[j]
            end
            push!(hp_set, hp)

        end
    end
    return hp_set
end

"""
    set_gamma!(hp_model::Dict)

Helper function to set Lipschitz bound hyperparameter
if required by the model type
"""
function set_gamma!(hp_model::Dict)

    # No need if not a γREN
    mtype = hp_model["modeltype"]
    if !occursin("gren", mtype)
        return nothing
    end

    # Assumes model type is like: "...-gren100-..."
    split_mtype = split(mtype,"-")
    indx = occursin.("gren", split_mtype)
    γ = parse(Float64, split_mtype[indx][1][5:end])
    hp_model["γ"] = γ

    return nothing

end

"""
    get_backup_control(G, Lb, Nb, Lo = nothing)

Construct backup (and optimal) controller for linear
systems, assuming diagonal covariance and cost matrices
with weights Nb and Lb (respectively)
"""
function get_backup_control(G, Lb, Nb, Lo = nothing)

    nx, nu, ny = G.nx, G.nu, G.ny

    # Backup controller
    Wb = diagm(0 => Nb[1:nx]) 
    Vb = diagm(0 => Nb[nx+1:nx+ny])
    Qb = diagm(0 => Lb[1:nx]) 
    Rb = diagm(0 => Lb[nx+1:nx+nu])
    Cb = OutputFeedback(G,Wb,Vb,Qb,Rb)

    (Lo === nothing) && (return Cb)

    # LQG controller
    Wo = (G.σw^2)*Matrix(I,nx,nx) 
    Vo = (G.σv^2)*Matrix(I,ny,ny)
    P0 = diagm(0 => G.x0_lims.^2)
    Qo = diagm(0 => Lo[1:nx]) 
    Ro = diagm(0 => Lo[nx+1:nx+nu])
    Co = LQG(G, Wo, Vo, Qo, Ro, P0)

    return Cb, Co

end

"""
Run random search algorithm
"""
function run_random_search(
    model, G, cost_func, x0, hp; 
    verbose = true,
    rng = Random.GLOBAL_RNG
)

    # Decide on training epochs
    n_epochs = Int(hp["n_evals"]/hp["policy_batches"])
    step_decay = hp["step_decay"] * n_epochs

    start = Dates.format(now(), "HH:MM")
    rewards, _ = random_search(
        model, G, cost_func; 
        n_epochs       = n_epochs,
        policy_batches = hp["policy_batches"],
        state_batches  = hp["state_batches"],
        explore_mag    = hp["explore_mag"],
        stepsize       = hp["stepsize"],
        max_grad_norm  = hp["grad_clip"],
        step_decay_ep  = step_decay,
        schedule_type  = hp["schedule_type"],
        rewardfreq     = hp["reward_freq"],
        verbose        = verbose,
        printfreq      = hp["printfreq"],
        x0_test        = x0,
        rng            = rng
    )
    costs = -rewards
    finish = Dates.format(now(), "HH:MM")
    verbose && println("Started: ",start,"\nFinished: ", finish)

    return costs, (start, finish)

end

"""
Save a model and training data
"""
function save_trained_model(
    exp_ps::ExperimentParams,
    model, 
    costs, 
    label,
    tspan
)

    hp = exp_ps.hyperparams
    fpath = string(exp_ps.savedir,label,".bson")
    bson(
        fpath, 
        Dict(
            "ModelType"     => hp["modeltype"],
            "FilterType"    => hp["filtertype"],
            "Model"         => model, 
            "Plant"         => exp_ps.env,
            "Hyperparams"   => hp,
            "Costs"         => costs,
            "BaseCosts"     => exp_ps.base_costs,
            "x0"            => exp_ps.x0_test,
            "Times"         => tspan,
        )
    )

end

"""
    plot_costs(label, costs, hp, Jb, Jo=nothing)

Plot costs vs traning epochs given vector of costs
"""
function plot_costs(label, costs, hp, Jb, Jo=nothing)

    # Sort out x-axis
    x = (0:length(costs)-1)*hp["reward_freq"] .+ 1

    # Plot
    f = Figure()
    ax = Axis(f[1,1], xlabel = "Epochs", ylabel = "Cost", title=label)
    lines!(ax, x, costs, color=:black, label="Model")
    lines!(ax, [1, x[end]], [Jb, Jb], color=:red, linestyle=:dash, label="Backup")
    (Jo === nothing) || lines!(ax, [1, x[end]], [Jo, Jo], color=:green, linestyle=:dash, label="Optimal")
    axislegend(position = :rt)
    display(f)

end

"""
    plot_costs(fpath::String)

Plot costs vs. training epochs given file path to saved data
"""
function plot_costs(fpath::String)
    label = split(fpath,"/")[end]
    data = BSON.load(fpath)
    plot_costs(label, data["Costs"], data["Hyperparams"], data["BaseCosts"]...)
    println(data["Costs"][end])
end

"""
    get_label(i::Int, e::ExperimentParams)

Construct a label for an experiment. Convention is:

    <model_type>_α_σ_N_Ns_nx_nv_ftype_<div_factor>_<grad_clip>_vX.bson
"""
function get_label(i::Int, e::ExperimentParams)

    hp = e.hyperparams
    version = string("v$(i-1)")

    label = string(
        hp["modeltype"],                "_",
        hp["filtertype"],               "_",
        my_float2str(hp["stepsize"]),   "_",
        my_float2str(hp["explore_mag"]),"_",
        hp["policy_batches"],           "_",
        hp["state_batches"],            "_",
        hp["nx"],                       "_",
        hp["nv"],                       "_",
        hp["div_factor"],               "_",
        hp["grad_clip"],                "_",
        version
    )

    return label
end

"""
    run_experiment_manual_loop(
        exp_ps::ExperimentParams; 
        verb_cost    = true, 
        verbose      = false, 
        plotting     = false, 
        dosave       = true, 
        return_model = false,
        rng          = Random.GLOBAL_RNG
    )

Run an experiment given an instance of `ExperimentParams`. This is
a newer version of the old `run_experiment()` function. It assumes
you loop over random seeds for `hp["n_experiments"]` yourself. Allows
more flexibility for distributed programming.
"""
function run_experiment_manual_loop(
    exp_ps::ExperimentParams; 
    verb_cost    = true, 
    verbose      = false, 
    plotting     = false, 
    dosave       = true, 
    return_model = false,
    rng          = Random.GLOBAL_RNG
)

    # Extract experiment details
    G           = exp_ps.env
    Gc          = exp_ps.env_model
    x0          = exp_ps.x0_test
    Cb          = exp_ps.backup
    hp          = exp_ps.hyperparams
    model_init  = exp_ps.model_init
    Jopt        = exp_ps.base_costs
    e           = exp_ps.version_num

    Jb = Jopt[1]
    (length(Jopt) == 2) ? (Jo = Jopt[2]) : (Jo = nothing)

    # Get experiment label
    label = get_label(e,exp_ps)

    # Train a model with random search
    model        = get_model(Gc, Cb, hp, model_init; rng=rng)
    costs, tspan = run_random_search(model, G, cost_func, x0, hp; verbose=verbose, rng=rng)

    # Save, plot, and print (where required)
    dosave    && save_trained_model(exp_ps, model, costs, label, tspan)
    plotting  && plot_costs(label, costs, hp, Jb, Jo)
    verb_cost && @printf("%-50s %.2f\n",label,costs[end])

    # Return a model if required
    return_model && (return model)
end

# Useful helper functions
load_costs(fname) = BSON.load(fname)["Costs"]
load_nevals(fname) = BSON.load(fname)["Hyperparams"]["n_evals"]

function load_hyperparam(fname::String, hp::String) 
    hp_dict = BSON.load(fname)["Hyperparams"]
    out = haskey(hp_dict, hp) ? hp_dict[hp] : NaN
    return out
end
load_hyperparam(fname::Vector, hp::String) = load_hyperparam.(fname, (hp,))

"""
    process_files(fpath)

Given a file path, read in important information from
all files in the directory and store in a big table 
for ease of use
"""
function process_files(fpath)

    # Get files and version numbers
    fnames = [string(fpath,fname) for fname in readdir(fpath)]
    fname_data = split.(fnames,"_")
    versions = [fname_data[i][end][1:2] for i in eachindex(fname_data)]

    # Fill in a table with notable hyperparameters
    tab_out = Table(
        fname = fnames,

        model = load_hyperparam(fnames, "modeltype"),
        filter = load_hyperparam(fnames, "filtertype"),
        α = load_hyperparam(fnames, "stepsize"),
        σ = load_hyperparam(fnames, "explore_mag"),
        N = load_hyperparam(fnames, "policy_batches"),
        Ns = load_hyperparam(fnames, "state_batches"),
        γ = load_hyperparam(fnames, "γ"),
        nx = load_hyperparam(fnames, "nx"),
        nv = load_hyperparam(fnames, "nv"),
        div_fact = load_hyperparam(fnames, "div_factor"),
        grad_clip = load_hyperparam(fnames, "grad_clip"),

        version = versions,
        costs = load_costs.(fnames),
        nevals = load_nevals.(fnames)
    )

    Jopt = BSON.load(fnames[1])["BaseCosts"]

    return tab_out, Jopt

end

# Helper function for plot legends
function get_legend_string(x)
    nx = Int(floor(log10(x)))
    x1 = Int(x*(10^(-nx)))
    return x1, nx
end

# Helper function to get cost statistics
function cost_stats(costs)
    μ = mean(costs)
    σ = std(costs)

    cs = zeros(length(costs), length(μ))
    for k in eachindex(costs)
        cs[k,:] = costs[k]
    end
    cmin = vec(minimum(cs;dims=1))
    cmax = vec(maximum(cs;dims=1))

    return μ, σ, cmin, cmax
end

# Helper functions for plot indexing
_r(i,n=3) = Int(ceil(i/n))
_c(i,n=3) = (i-1)%n + 1