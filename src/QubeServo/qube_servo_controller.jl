using MatrixEquations

"""
$(TYPEDEF)

Energy-pumping swingup controller for non-linear Qube Servo 2.
"""
mutable struct QubeSwingUp <: AbstractCtrlModel
    nx::Int
    G::QubeServo2
    K
    Gain
    Threshold
    xref
end

function QubeSwingUp(G::QubeServo2, K; k=600, threshold=30*π/180)
    QubeSwingUp(0, G, K, k, threshold, [0, π, 0, 0])
end

"""
    QubeSwingUp(G::QubeServo2, A, B, Q, R; k=100, threshold=30*π/180)

Design swingup + LQR controller given linear
state-space matrices A, B and quadratic cost
matrices Q and R on position/control (respectively).

LQR controller takes over at the `threshold`. Gain `k`
is the swingup controller gain.
"""
function QubeSwingUp(G::QubeServo2, A, B, Q, R; k=600, threshold=30*π/180)

    # Get LQR controller gain
    S2 = zeros(G.nx, G.nu)
    _, _, K, _ = ared(A,B,R,Q,S2)

    return QubeSwingUp(0, G, K, k, threshold, [0, π, 0, 0])
end

"""
    (c::BackstepMagLevCtrl)(x̂::AbstractVecOrMat)

Call backstepping model to compute control action given 
current state estimate x̂ of the system.

Controller written without array mutation for Flux.
"""
function (c::QubeSwingUp)(x::AbstractVecOrMat)

    # Useful variables
    g = c.G.g

    mp = c.G.mp
    Lp = c.G.Lp
    Jp = mp*Lp^2/12

    Mr = c.G.Mr
    Lr = c.G.Lr
    Rm = c.G.Rm
    kt = c.G.kt

    # From controller
    K = c.K
    k = c.Gain
    thresh = c.Threshold

    # Get α and αd from the state estimate, wrap angles
    x̂ = deepcopy(x)
    if typeof(x̂) <: AbstractMatrix
        x̂ = [wrap_angles(x̂[1:1,:], -π, π);
             wrap_angles(x̂[2:2,:]); x̂[3:4,:]]
        α = x̂[2:2,:]
        αd = x̂[4:4,:]
    elseif typeof(x̂) <: AbstractVector
        x̂ = [wrap_angles(x̂[1], -π, π);
             wrap_angles(x̂[2]); x̂[3:4]]
        α = x̂[2:2]
        αd = x̂[4:4]
    end

    # Edge case when pendulum in downwards equilibrium
    indx = (α .≈ 0) .&& (αd .≈ 0)
    αd = αd .* (.!indx) + 1e-3 * indx

    # Feedback control with LQR, including linear offsets
    u_fb = -K * (x̂ .- c.xref)

    # Energy swingup
    cosα        = cos.(α)
    Eref        = mp * Lp * g
    energy      = 0.5*(Jp * αd.^2 .+ mp * Lp * g * (1 .- cosα))
    u_energy    = (energy .- Eref) .* sign.(αd .* cosα)
    u_energy   *= k * (Mr * Lr * Rm / kt)

    # Combine them where appropriate
    indx       = abs.(α .- π) .<= thresh
    u = u_fb .* indx + u_energy .* (.!indx) 

    # Clamp the output
    u = clamp.(u, -c.G.v_max, c.G.v_max)

    return u

end 

"""
    (c::QubeSwingUp)(ξ::AbstractVecOrMat, x̂, u=0, t=0; rng=nothing)

Call method for compatibility with other `AbstractCtrlModel`s.
Internal state ξ is empty, and time t is unused.
"""
function (c::QubeSwingUp)(ξ::AbstractVecOrMat, x̂, t=0; rng=nothing)
    return ξ, c(x̂)
end

"""
    control_action(c::QubeSwingUp, ξ, x̂, t=0)

Compute control action. Just calls the controller model itself.
"""
function control_action(c::QubeSwingUp, x̂, y, t=0)
    return c(x̂)
end

"""
$(TYPEDEF)

Output-feedback controller for Qube servo system.
Consists of a high-gain observer and the swingup
controller + LQR to stabilise the arm in the upright
position.
"""
mutable struct QubeCtrl <: AbstractCtrlModel
    nx::Int                 # Number of observer states
    L::AbstractMatrix       # Observer gain matrix
    ctrl::QubeSwingUp       # State-feedback controller
end

"""
    QubeCtrl(
        G::QubeServo2,
        A, B, Q, R;
        α1 = 1, α2 = 1, ϵ = G.Δt,
        k=600, threshold=30*π/180
    )

Constructor for the output-feedback controller.
"""
function QubeCtrl(
    G::QubeServo2,
    A, B, Q, R;
    α1 = 1, α2 = 1, ϵ = G.Δt,
    k=600, threshold=30*π/180
)
    nx = G.nx
    return QubeCtrl(
        nx,
        [α1/ϵ 0; 0 α2/ϵ; α1/ϵ^2 0; 0 α2/ϵ^2],
        QubeSwingUp(G, A, B, Q, R; k=k, threshold=threshold)
    )
end

"""
    (m::QubeCtrl)(x̂, y, u=0, t=0; rng=nothing)

Call the output feedback system to compute controls and
update the observer state
"""
function (c::QubeCtrl)(x̂, y, u=0, t=0; rng=nothing)

    # Observer and control updates
    u = control_action(c, x̂, y)
    x̂ = observer_update(c, x̂, y, u, 0; rng=rng)

    return x̂, u
end

"""
    control_action(c::QubeCtrl, x̂, y=0, t=0)

Control action for `QubeCtrl` just calls controller for
`QubeSwingUp` directly using the state estimate
"""
control_action(c::QubeCtrl, x̂, y, t=0) = c.ctrl(x̂)

"""
    cts_observer(m::QubeCtrl, x̂, ỹ, u)

Continuous observer dynamics
"""
function cts_observer(m::QubeCtrl, x̂, y, u)
    ỹ = y - measure(m.ctrl.G, x̂)
    return ct_dynamics(m.ctrl.G, x̂, u) + m.L * ỹ
end

"""
    observer_update(m::QubeCtrl, x̂, y, u, t; rng=...)

Run the `QubeCtrl` high-gain observer
"""
function observer_update(m::QubeCtrl, x̂, y, u, t; rng=nothing)

    # Useful
    Δt = m.ctrl.G.Δt

    # Rungw-Kutta 4 scheme
    k1 = cts_observer(m, x̂, y, u)
    k2 = cts_observer(m, x̂ + Δt*k1/2, y, u)
    k3 = cts_observer(m, x̂ + Δt*k2/2, y, u)
    k4 = cts_observer(m, x̂ + Δt*k3, y, u)
    x̂ = x̂ + Δt*(k1 + 2k2 + 2k3 + k4)/6

    return x̂

end


"""
$(TYPEDEF)

Separate high-gain observer for the Qube Servo environment
"""
mutable struct TestHighGain
    G::AbstractEnvironment
    L::AbstractMatrix
    nx::Int
end

function TestHighGain(G; ϵ=0.04)
    nx = 4
    L = [1/ϵ 0; 0 1/ϵ; 1/ϵ^2 0; 0 1/ϵ^2]
    return TestHighGain(G, L, nx)
end

function cts_observer(m::TestHighGain, x̂, y, u)
    ỹ = y - measure(m.G, x̂)
    return ct_dynamics(m.G, x̂, u) + m.L * ỹ
end

function (m::TestHighGain)(x̂, y, u)

    # Useful
    Δt = m.G.Δt

    # Rungw-Kutta 4 scheme
    k1 = cts_observer(m, x̂, y, u)
    k2 = cts_observer(m, x̂ + Δt*k1/2, y, u)
    k3 = cts_observer(m, x̂ + Δt*k2/2, y, u)
    k4 = cts_observer(m, x̂ + Δt*k3, y, u)
    x̂ = x̂ + Δt*(k1 + 2k2 + 2k3 + k4)/6

    return x̂

end

observer_update(m::TestHighGain, x̂, y, u, t; rng=nothing) = m(x̂, y, u)

init_state(m::TestHighGain) = zeros(m.nx)
init_state(m::TestHighGain, batches::Int) = zeros(m.nx, batches)