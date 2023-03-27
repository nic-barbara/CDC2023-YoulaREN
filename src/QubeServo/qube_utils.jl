"""
    wrap_angles(x)

Helper function to wrap angles (in rad)
between 0 and 2π
"""
function wrap_angles(x::Number, lower=0, upper=2π)
    if x < lower
        return x + ceil(abs(x)/2π)*2π
    elseif x > upper
        return x - floor(abs(x)/2π)*2π
    else
        return x
    end
end
wrap_angles(x::AbstractVecOrMat, lower=0, upper=2π) = wrap_angles.(x, lower, upper)


function sample_qube2_space(
    G::QubeServo2, model, cost_func=(x,u) -> 0; 
    nsims=100, rng=StableRNG(4),
    title = "",
    in_axes = nothing,
    show_fig = true,
    colour = :grey,
    lwidth = 0.5
)

    # Useful
    dt = G.Δt
    ts = dt:dt:G.max_steps*dt

    # Initialise 
    _z() = zeros(nsims, G.max_steps)
    θ = _z()
    α = _z()
    u = _z()
    costs = zeros(nsims)

    # Simulate
    for k in 1:nsims
        x0 = init_state(G; rng=rng)
        Js, _, us, ys = rollout(G, model, cost_func, true; x0=x0, rng=rng)
        θ[k,:] = ys[1,:]
        α[k,:] = ys[2,:]
        u[k,:] = us[1,:]
        costs[k] = Js[end]
    end

    # Convert to deg
    θ .= rad2deg.(θ)
    α .= rad2deg.(α)

    # Construct figure
    if in_axes === nothing
        size_inches = (8, 8)
        size_pt = 100 .* size_inches
        f = Figure(resolution = size_pt, fontsize = 16)
        ga = f[1,1] = GridLayout()

        ax1 = Axis(ga[1,1], xlabel = "Time (s)", ylabel = "Arm angle (deg)", title=title)
        ax2 = Axis(ga[1,2], xlabel = "Time (s)", ylabel = "Pendulum angle (deg)")
        ax3 = Axis(ga[2,1:2], xlabel = "Time (s)", ylabel = "Control (V)")
    else
        ax1, ax2, ax3, f = in_axes
    end
    
    # Plot everything
    for k in axes(θ,1)
        lines!(ax1, ts, θ[k,:], linewidth=lwidth, color=colour)
        lines!(ax2, ts, α[k,:], linewidth=lwidth, color=colour)
        lines!(ax3, ts, u[k,:], linewidth=lwidth, color=colour)
    end
    ylims!(ax3, -20.0, 20.0)

    show_fig && display(f)

    return mean(costs), (ax1, ax2, ax3, f)

end

# Plot simulations from a selected model
function plot_qube_sim(data, model_type, cost_func=(x,u) -> 0; rng=StableRNG(4))

    # Find model with best final cost
    d = data[(data.model .== model_type)]
    J = get_cost.(d.costs)
    indx = (1:length(d))[J .== minimum(J)][1]

    # Get the data
    model_data = BSON.load(d[indx].fname)
    model = model_data["Model"]
    G = model_data["Plant"]

    # Plot the sim
    println(model_type)
    if !(model_type == "ren" || model_type == "lstm")
    _, ax = sample_qube2_space(
        G, model.C, cost_func; show_fig=false,
        title=model_type, rng=rng
    )
    J1, _ = sample_qube2_space(
        G, model, cost_func; in_axes=ax, colour=:red, lwidth=0.25, rng=rng
    )
    else
        J1, _ = sample_qube2_space(G, model, cost_func; title=model_type, rng=rng)
    end

    (J1 != 0) && (return J1)
    return minimum(J)
end

get_cost(x) = x[end]
