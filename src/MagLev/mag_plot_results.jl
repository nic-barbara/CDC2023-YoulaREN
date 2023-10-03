cd(@__DIR__)
using Pkg
Pkg.activate("./../..")

using BSON
using CairoMakie
using Revise
using StableRNGs

includet("../../utils/load_utils.jl")
includet("./../Robustness/utils.jl")
includet("./output_fdback_backstep.jl")
includet("./mag_utils.jl")

# Legacy fix
RecurrentEquilibriumNetworks = RobustNeuralNetworks

do_save = true

# Read in data
fpath = "../../results/mag-experiment/"
data, Jb = process_files(fpath)
Jb = Jb[1]
J_nmpc = 29.457 / Jb

# Figure setup
fsize = 24
size_inches = (8, 6)
size_pt = 100 .* size_inches

f = Figure(resolution = size_pt, fontsize = fsize)
ga = f[1,1] = GridLayout()
ax = Axis(ga[1,1], xlabel = "Epochs", ylabel = "Normalized cost")

# Models and x-axis array
models = unique(data.model)
models, n_nonlip = order_files(models)
nc = length(data.costs[1])
x = (0:nc-1)/nc*data.nevals[1]/data.N[1]

function plot_results()

    # Loop models and plot
    for k in eachindex(models)

        m = models[k]
        lab = get_label(m)
        lstyle = get_lstyle(m)
        colour = get_colour(m, k, n_nonlip)

        m_costs = data[data.model .==  models[k]].costs/Jb
        μ, _, cmin, cmax = cost_stats(m_costs)

        band!(ax, x, cmax, cmin, color = (colour, 0.2))
        lines!(ax,x, μ, label=lab, linewidth=2, color=colour, linestyle=lstyle)

    end

    # Add a line for NMPC and base controller
    lines!(ax, [1, x[end]], [J_nmpc, J_nmpc], color=:red, linewidth=2, label="NMPC")
    lines!(ax, [1, x[end]], [1.0, 1.0], color=:black, linewidth=2)

    # Formatting
    xlims!(ax,(0.0,x[end]))
    ylims!(ax,(0.6,1.05))

    Legend(ga[2,1], ax, orientation=:horizontal, nbanks=3)
    display(f)

    do_save && save("mag_learning.pdf", f)
    return nothing

end

plot_results()
