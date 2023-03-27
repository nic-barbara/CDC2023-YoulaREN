"""
$(TYPEDEF)

LSTM model written by Ray
"""
mutable struct lstm <: AbstractCtrlModel
    nu::Int
    nv::Int
    ny::Int
    nx::Int                 # Number of states = 2*nv for LSTM
    A::Matrix{Float64}
    B::Matrix{Float64}
    C::Matrix{Float64}
    bx::Vector{Float64}
    by::Vector{Float64}
end

"""
    lstm(nu::Int, nv::Int, ny::Int)

Initialise LSTM network from given
input, hidden, and output sizes.
"""
function lstm(nu::Int, nv::Int, ny::Int; rng=Random.GLOBAL_RNG)    
    nx = 4 * nv 
    A = glorot_normal(nx, nv; rng=rng)
    B = glorot_normal(nx, nu; rng=rng)
    C = glorot_normal(ny, nv; rng=rng)
    bx = zeros(nx)/sqrt(nx)
    by = zeros(ny)/sqrt(ny)
    bx[nv+1:2*nv] .= 1

    return lstm(nu, nv, ny, 2*nv, A, B, C, bx, by)
end

"""
    Flux.trainable(m::lstm)

Define trainable parameters
"""
Flux.trainable(m::lstm) = (m.A, m.B, m.C, m.bx, m.by)

"""
    (m::lstm)(ξ0, u, t=0)

Call LSTM given states and input
"""
function (m::lstm)(ξ0, u, t=0)
    xt = m.A * _hr(ξ0,1:m.nv) + m.B * u .+ m.bx
    ft = Flux.sigmoid.(_hr(xt, 1:m.nv))
    it = Flux.sigmoid.(_hr(xt, m.nv+1:2*m.nv))
    ot = Flux.sigmoid.(_hr(xt, 2*m.nv+1:3*m.nv))
    ct = Flux.tanh.(_hr(xt, 3*m.nv+1:4*m.nv))
    c  = ft .* _hr(ξ0, m.nv+1:2*m.nv) .+ it .* ct
    h  = ot .* Flux.tanh.(c)
    y  = m.C * h .+ m.by 

    return vcat(h,c), y 
end

# Helper function to pick rows
_hr(x::AbstractVector,rows) = x[rows]
_hr(x::AbstractMatrix,rows) = x[rows,:]

(m::lstm)(ξ, y, u, t; rng=nothing) = m(ξ, y, t)

"""
    set_output_zero!(m::lstm)

Extend RobustNeuralNetworks.jl method to set
the LSTM output to 0
"""
function RobustNeuralNetworks.set_output_zero!(m::lstm)
    m.C  .*= 0.0
    m.by .*= 0.0
    return nothing
end
