cd(@__DIR__)
using Pkg
Pkg.activate("./../..")

using BSON
using CairoMakie
using ColorSchemes
using RobustNeuralNetworks
using Revise
using StableRNGs

includet("utils.jl")

# Save?
do_save = true

# File path
fpath = "../../results/qube-robust/experiment/"
fnames = readdir(fpath)
ϵcross = 1.25
Jb = 1534

# Sort files nicely
fnames, n_nonlip = order_files(fnames)

# Find all files for same model
fnames_no_version_num = get_name.(fnames)
models = unique(fnames_no_version_num)

# Set up a figure
size_inches = (12, 5)
size_pt = 100 .* size_inches
f1 = Figure(resolution = size_pt, fontsize = 24)
ga = f1[1,1] = GridLayout()

# Set up the axes
ax1 = Axis(ga[1,1], xlabel="Attack Size", ylabel="Normalized Cost")
ax2 = Axis(ga[1,2], xlabel="Lipschitz Lower Bound", ylabel="Critical Attack Size",
           xticks=([2,3,4,5],["10²", "10³", "10⁴", "10⁵"]))

# Add a line for the cross-section
xmin, xmax = 0.0, 0.8
ymin, ymax = 0.5, 1.75
# lines!(ax1, [xmin, xmax], [ϵcross, ϵcross], color=:brown, linestyle=:dot)

# Plot options
alpha  = 0.5
wwidth = 10
msize  = 20
lwidth = 2

# Plot average result over all random seeds
for k in eachindex(models)

    m = models[k]
    colour = get_colour(m, k, n_nonlip; fudge_colours=true)
    marker = get_marker(m)
    lstyle = get_lstyle(m)
    bar_colour = colour

    # Arrays to log data
    γmax_vec = Vector{Float64}([])
    γmin_vec = Vector{Float64}([])
    costs_vec = Vector{Vector{Float64}}([])
    ϵcrit_vec = Vector{Float64}([])
    ϵs = nothing

    model_files = fnames[m .== fnames_no_version_num]
    for fname in model_files

        # Load data
        data = BSON.load(string(fpath,fname))
        push!(γmax_vec, data["γmax"])
        push!(γmin_vec, data["γmin"])
        push!(costs_vec, data["costs"] ./ Jb)
        ϵs = data["ϵs"]

        # Load ϵcrit separately for now
        epath = "../../results/qube-robust/ecrit/"
        ecrit_name = string("ecrit", fname[4:end])
        edata = BSON.load(string(epath, ecrit_name))
        push!(ϵcrit_vec, edata["ϵcrit"])

    end

    # Get min/max/mean of each
    γmax_mean, γmax_min, γmax_max = get_stats(γmax_vec)
    γmin_mean, γmin_min, γmin_max = get_stats(γmin_vec)
    costs_mean, costs_min, costs_max = get_stats(costs_vec)
    ϵcrit_mean, ϵcrit_min, ϵcrit_max = get_stats(ϵcrit_vec)

    # Make a label for plotting
    lab = get_label(m)

    # Plot cost as function of attack size with band
    band!(ax1, ϵs, costs_max, costs_min, color = (colour, 0.2))
    lines!(ax1, ϵs, costs_mean, label = lab, color = colour, 
           linewidth=lwidth, linestyle = lstyle)

    # Plot critical eps as scatter plot with error bars
    scatter!(
        ax2, log10(γmin_mean), ϵcrit_mean, marker = marker, 
        markersize = msize, label = lab, color = colour
    )
    rangebars!(
        ax2, [log10(γmin_mean)], [ϵcrit_min], [ϵcrit_max], 
        color=bar_colour, direction=:y, whiskerwidth=wwidth
    )
    rangebars!(
        ax2, [ϵcrit_mean], [log10(γmin_min)], [log10(γmin_max)], 
        color=bar_colour, direction=:x, whiskerwidth=wwidth
    )

end

# Format
xlims!(ax1, xmin, xmax)
ylims!(ax1, ymin, ymax)

xlims!(ax2, log10(20), 4.1)
ylims!(ax2, 0.3, 0.8)


Legend(ga[2,1:2], ax1, orientation=:horizontal,nbanks=2)
display(f1)

# Save the figure
do_save && save("qube_robustness_ecrit.pdf", f1)
