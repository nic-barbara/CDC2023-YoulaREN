############ Package dependencies ############

using ControlSystems
using DocStringExtensions
using LinearAlgebra
using MAT
using Random
using Revise
using StableRNGs


############ Abstract types ############

"""
$(TYPEDEF)
"""
abstract type AbstractEnvironment end

"""
$(TYPEDEF)
"""
abstract type LTI <: AbstractEnvironment end


############ Includes ############

includet("./utils.jl")
includet("./lti.jl")

includet("./qube_servo2.jl")
includet("./maglev.jl")


############ Other ############
