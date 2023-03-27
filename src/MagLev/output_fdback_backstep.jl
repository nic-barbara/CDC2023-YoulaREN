using MatrixEquations

"""
$(TYPEDEF)

Nonlinear backstepping controller designed for the magnetic suspension
system presented in Khalil, Nonlinear Systems (3rd Ed.) 2002.

See solutions to Exercise 14.39 (a) and (b) for a derivation.
"""
mutable struct BackstepMagLevCtrl <: AbstractCtrlModel
    nx :: Int
    G :: MagLevEnv
    yref :: Number
    P :: AbstractMatrix
    k1 :: Number
    k2 :: Number
    γ0 :: Number
    γ1 :: Number
end

"""
    BackstepMagLevCtrl(...)

Default controller constructor. Specify target position
for the ball as `yref`, and define control gains.
"""
function BackstepMagLevCtrl(
    G :: MagLevEnv;
    yref :: Number = 0.05,
    k1 :: Number = 100,     # Proportional gain on position
    k2 :: Number = 1,       # Gain on velocity (leave it fairly unregulated)
    γ0 :: Number = 100,     # Less oscillatory control signal (like D gain)
    γ1 :: Number = 5        # Faster to target, but more overshoot (like P gain)
)

    # Compute solution to Lyapunov equation
    A = [0 1; -k1/G.m -(G.k+k1)/G.m]
    P = lyapc(A', I(2))

    # Construct the controller
    return BackstepMagLevCtrl(0, G, yref, P, k1, k2, γ0, γ1)
end

"""
    (c::BackstepMagLevCtrl)(x̂::AbstractVecOrMat)

Call backstepping model to compute control action given 
current state estimate x̂ of the system.
"""
function (c::BackstepMagLevCtrl)(x̂::AbstractVecOrMat)

    # Useful variables
    m = c.G.m
    g = c.G.g
    a = c.G.a
    k = c.G.k
    R = c.G.R
    L0 = c.G.L0
    L1 = c.G.L1

    # From controller
    r = c.yref
    k1 = c.k1
    k2 = c.k2
    γ0 = c.γ0
    γ1 = c.γ1
    
    # Entries in P matrix
    p3 = c.P[3]
    p4 = c.P[4]

    # Pull out the states
    if typeof(x̂) <: AbstractMatrix
        x1 = x̂[1:1,:] .- r
        x2 = x̂[2:2,:]
        x3 = x̂[3:3,:]
    elseif typeof(x̂) <: AbstractVector
        x1 = x̂[1:1] .- r
        x2 = x̂[2:2]
        x3 = x̂[3:3]
    end

    # Useful terms
    arx1 = (a + r) .+ x1
    w = 2*(x1*p3 + x2*p4)/m
    Lx1 = L1 .+ a * L0 ./ arx1
    F = -(L0 * a * x3.^2) ./ (2*arx1.^2)
    ϕ = -(k1*x1 - k2*x2 - γ0*w) .- m*g
    z = F - ϕ

    # Compute derivatives for coefficient expressions
    dFdx1 = (L0 * a * x3.^2) ./ (arx1.^3)
    dFdx3 = -(L0 * a * x3) ./ (arx1.^2)

    dϕdx1 = -k1 - 2γ0*p3/m
    dϕdx2 = -k2 - 2γ0*p4/m

    # Compute coefficient expressions
    α1 = (dFdx1 .- dϕdx1) .* x2 +
         (dFdx3 .* (-R*x3 + (L0 * a * x2 .* x3 ./ arx1.^2)) ./ Lx1) - 
         (dϕdx2 .* (g .+ (F - k * x2) ./ m) )
    α2 = dFdx3 ./ Lx1

    # Compute control signal
    u = -(α1 + w + γ1*z) ./ α2

    # Clamp it as necessary
    return clamp.(u, c.G.v_min, c.G.v_max)

end

"""
    (c::BackstepMagLevCtrl)(ξ::AbstractVecOrMat, x̂, u=0, t=0; rng=nothing)

Call method for compatibility with other `AbstractCtrlModel`s.
Internal state ξ is empty, and time t is unused.
"""
function (c::BackstepMagLevCtrl)(ξ::AbstractVecOrMat, x̂, u=0, t=0; rng=nothing)
    return ξ, c(x̂)
end

"""
    control_action(c::BackstepMagLevCtrl, ξ, x̂, t=0)

Compute control action. Just calls the controller model itself.
"""
function control_action(c::BackstepMagLevCtrl, x̂, y, t=0)
    return c(x̂)
end

"""
$(TYPEDEF)

Output-feedback controller for maglev system.
Consists of a high-gain observer and backstep
state-feedback controller.
"""
mutable struct MagLevCtrl <: AbstractCtrlModel
    nx::Int                     # Number of observer states
    L::AbstractMatrix           # Observer gain matrix
    ctrl
end

"""
    MagLevCtrl(G::MagLevEnv; α1 = 1, α2 = 1, ϵ = G.Δt,
        yref = 0.05, k1 = 100, k2 = 1, γ0 = 100, γ1 = 5)

Construct controller. Note that the observer only 
estimates position and velocity, assuming that current 
can be measured reasonably accurately.
"""
function MagLevCtrl(
    G::MagLevEnv;
    α1 = 1, α2 = 1, ϵ = G.Δt,
    yref = 0.05, k1 = 100, k2 = 1, γ0 = 100, γ1 = 5
)
    nx = 3
    return MagLevCtrl(
        nx,
        reshape([α1/ϵ, α2/ϵ^2], nx-1, 1),
        BackstepMagLevCtrl(G; yref=yref,
            k1=k1, k2=k2, γ0=γ0, γ1=γ1)
    )
end

"""
    (m::MagLevCtrl)(x̂, y, u=0, t=0; rng=nothing)

Call the output feedback system to compute controls and
update the observer state
"""
function (c::MagLevCtrl)(x̂, y, u=0, t=0; rng=nothing)

    u = control_action(c, x̂, y)
    x̂ = observer_update(c, x̂, y, u; rng=rng)

    return x̂, u
end

"""
    set_obsv_state(x̂::AbstractVector,y::AbstractVector)

Set third entry of observer state to the measured value
i.e: x̂[3] = y[2]. Written for no array mutation
"""
set_obsv_state(x̂::AbstractVector,y::AbstractVector) = [x̂[1], x̂[2], y[2]]
set_obsv_state(x̂::AbstractMatrix,y::AbstractMatrix) = [x̂[1:1,:]; x̂[2:2,:]; y[2:2,:]]


"""
    control_action(c::MagLevCtrl, x̂, y=0, t=0)

Control action for `MagLevCtrl` just calls controller for
`BackstepMagLevCtrl` directly after adding x3 to the state
estimate from the observer
"""
function control_action(c::MagLevCtrl, x̂, y, t=0)
    x̂ = set_obsv_state(x̂,y)
    return c.ctrl(x̂)
end

"""
    cts_observer(m::MagLevCtrl, x̂, ỹ, u)

Continuous observer dynamics
"""
function cts_observer(m::MagLevCtrl, x̂, y, u)
    w = m.L * (y - [1 0 0] * x̂)
    w = vcat(w, zeros(size(y)))
    return ct_dynamics(m.ctrl.G, x̂, u) + w
end

"""
    observer_update(m::MagLevCtrl, x̂, y, u, t=0; rng=...)

Run the `MagLevCtrl` high-gain observer

Assumes inputs are `x̂ = [x̂1, x̂2, x̂3]` (state of the observer) 
and `y = [x1, x3]` (measurements of the whole system). 

The observer uses `y` directly to get `x̂3 = y[2]` and only
estimates `x̂ = [x̂1, x̂2]` using `y[1]`. No estimation of x̂[3]
is performed (but the variable will change!). This is accounted
for by the controller.
"""
function observer_update(m::MagLevCtrl, x̂, y, u, t=0; rng=nothing)

    # Useful
    Δt = m.ctrl.G.Δt

    # Make sure xhat[3] is correct
    x̂ = set_obsv_state(x̂,y)

    # Innovations used by observer, ỹ[1]
    y = pick_rows(y, 1:1)

    # Runge-Kutta 4 scheme
    k1 = cts_observer(m, x̂, y, u)
    k2 = cts_observer(m, x̂ + Δt*k1/2, y, u)
    k3 = cts_observer(m, x̂ + Δt*k2/2, y, u)
    k4 = cts_observer(m, x̂ + Δt*k3, y, u)
    x̂ = x̂ + Δt*(k1 + 2k2 + 2k3 + k4)/6

    return x̂

end

init_state(m::MagLevCtrl) = [m.ctrl.yref, 0, 0]
init_state(m::MagLevCtrl, batches) = [m.ctrl.yref*ones(1, batches); zeros(2,batches)]

# Useful
pick_rows(x::AbstractVector, rows) = x[rows]
pick_rows(x::AbstractMatrix, rows) = x[rows,:]



# TODO: This needs to be update. Code is hacky for now,
#       but it works fine. Can be properly streamlined later. 

"""
$(TYPEDEF)
"""
mutable struct MagLevObsv
    G::MagLevEnv
    nx::Int                     # Number of observer states
    L::AbstractMatrix           # Observer gain matrix
    yref
end

"""
"""
function MagLevObsv(
    G::MagLevEnv;
    α1 = 1, α2 = 1, ϵ = G.Δt,
    yref = 0.05
)
    nx = 3
    return MagLevObsv(G, nx, reshape([α1/ϵ, α2/ϵ^2], nx-1, 1), yref)
end

"""
"""
function cts_observer(m::MagLevObsv, x̂, y, u)
    w = m.L * (y - [1 0 0] * x̂)
    w = vcat(w, zeros(size(y)))
    return ct_dynamics(m.G, x̂, u) + w
end

"""
"""
function (m::MagLevObsv)(x̂, y, u)

    # Useful
    Δt = m.G.Δt

    # Make sure xhat[3] is correct
    x̂ = set_obsv_state(x̂,y)

    # Innovations used by observer, ỹ[1]
    y = pick_rows(y, 1:1)

    # Rungw-Kutta 4 scheme (adjusted for innovations as separate signal)
    k1 = cts_observer(m, x̂, y, u)
    k2 = cts_observer(m, x̂ + Δt*k1/2, y, u)
    k3 = cts_observer(m, x̂ + Δt*k2/2, y, u)
    k4 = cts_observer(m, x̂ + Δt*k3, y, u)
    x̂ = x̂ + Δt*(k1 + 2k2 + 2k3 + k4)/6

    return x̂

end

observer_update(m::MagLevObsv, x̂, y, u, t=0; rng=nothing) = m(x̂, y, u)

init_state(m::MagLevObsv) = zeros(3)
init_state(m::MagLevObsv, batches) = zeros(3, batches)
