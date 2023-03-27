############ Package dependencies ############

using ControlSystems: StateSpace
using DocStringExtensions
using Flux
using MatrixEquations
using Random
using RobustNeuralNetworks
using Revise
using StableRNGs


############ Abstract types ############

"""
$(TYPEDEF)
"""
abstract type AbstractCtrlModel end

"""
$(TYPEDEF)
"""
abstract type AbstractFilter end

"""
$(TYPEDEF)
"""
abstract type AbstractFeedback <: AbstractCtrlModel end


############ Includes ############
includet("./utils.jl")

includet("./filters.jl")
includet("./feedback_architectures.jl")
includet("./neural_networks.jl")


############ Other ############

# Generic tool for resetting dynamic policy. Use as you see fit
policy_reset!(m) = nothing
