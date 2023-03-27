function sample_maglev_space(
    G::MagLevEnv, model, cost_func; 
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
    pos = zeros(nsims, G.max_steps)
    vel = zeros(nsims, G.max_steps)
    ctrl = zeros(nsims, G.max_steps)
    costs = zeros(nsims)

    # Simulate
    for k in 1:nsims
        x0 = init_state(G; rng=rng)
        Js, xs, us, ys = rollout(G, model, cost_func, true; x0=x0, rng=rng)
        pos[k,:] = ys[1,:]
        vel[k,:] = ys[2,:]
        ctrl[k,:] = us[1,:]
        costs[k] = Js[end]
    end

    # Construct figure
    if in_axes === nothing
        size_inches = (8, 4)
        size_pt = 100 .* size_inches
        f = Figure(resolution = size_pt, fontsize = 16)
        ga = f[1,1] = GridLayout()

        ax1 = Axis(ga[1,1], xlabel = "Time (s)", ylabel = "Vertical height (m)", title=title)
        ax2 = Axis(ga[1,2], xlabel = "Time (s)", ylabel = "Control (V)")
    else
        ax1, ax2, f = in_axes
    end
    
    # Plot position
    for k in axes(pos,1)
        lines!(ax1, ts, pos[k,:], linewidth=lwidth, color=colour)
    end
    ylims!(ax1, 0.0, 0.1)
    ax1.yreversed = true

    # Plot control input
    for k in axes(pos,1)
        lines!(ax2, ts, ctrl[k,:], linewidth=lwidth, color=colour)
    end
    ylims!(ax2, 0.0, 15.0)

    show_fig && display(f)

    return mean(costs), (ax1, ax2, f)

end

# Plot simulations from a selected model
function plot_mag_sim(data, model_type, cost_func=(x,u) -> 0)

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
        _, ax = sample_maglev_space(
            G, model.C, cost_func; show_fig=false,
            title=model_type
        )
        J1, _ = sample_maglev_space(
            G, model, cost_func; in_axes=ax, colour=:red, lwidth=0.25
        )
    else
        J1, _ = sample_maglev_space(G, model, cost_func; title=model_type)
    end

    (J1 != 0) && (return J1)
    return minimum(J)
end

get_cost(x) = x[end]
