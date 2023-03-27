"""
$(TYPEDEF)

Nonlinear Qube Servo 2 dynamics
"""
mutable struct QubeServo2 <: AbstractEnvironment
    nx::Int64
    nu::Int64
    ny::Int64
    max_steps::Int64
    x0_lims::AbstractVector
    rng::AbstractRNG
    σw                                  # Process noise covariance is σw^2
    σv                                  # Measurement noise covariance is σv^2
    Rm                                  # Motor resistance (Ω)
    kt                                  # Motor current-torque (N-m/A)
    km                                  # Motor back-emf constant (V-s/rad)
    Mr                                  # Rotary arm mass (kg)
    Lr                                  # Rotary arm total length (m)
    Jr                                  # Rotary arm moment of inertia about pivot (kg-m^2)
    Dr                                  # Rotary arm equivalent viscous damping coefficient (N-m-s/rad)
    mp                                  # Pendulum link mass (kg)
    Lp                                  # Pendulum link total length (m)
    Jp                                  # Pendulum link moment of inertia about pivot (kg-m^2)
    Dp                                  # Pendulum link equivalent viscous damping coefficient (N-m-s/rad)
    g                                   # Gravity constant (m/s^2)
    Δt                                  # Sample time (s)
    v_max                               # Maximum voltage (V)
end

"""
    QubeServo2(...)

Construct Qube Servo 2 environment with sample time `Δt`,
maximum simulation steps `max_steps`, initial condition range
`x0_lims`, and random seed `rng`.
"""
function QubeServo2(;
    Δt = 0.02,
    max_steps = 150,
    x0_lims = [0.5,0.5,0.2,0.2],
    rng = StableRNG(0)
)

    # Motor
    Rm = 8.4
    kt = 0.042
    km = 0.042

    # Rotary Arm
    Mr = 0.095
    Lr = 0.085
    Jr = Mr*Lr^2/12
    Dr = 0.0015
 
    # Pendulum Link
    mp = 0.024
    Lp = 0.129
    Jp = mp*Lp^2/12
    Dp = 0.0005
    g = 9.81

    # Parallel axis theorem
    Jr = Jr + 0.25*Mr*Lr^2
    Jp = Jp + 0.25*mp*Lp^2

    # Noise
    σw = 1e-2
    σv = 1e-2

    # Voltage
    v_max = 20

    # Sizes
    nx, nu, ny = 4, 1, 2

    # Construct model
    QubeServo2(
        nx, nu, ny, max_steps, x0_lims, rng, σw, σv,
        Rm, kt, km, Mr, Lr, Jr, Dr, mp, Lp, Jp, Dp, g, Δt, v_max
    )

end

"""
    measure(G::QubeServo2, xt::AbstractVector, ...)

Compute measurement given current state. Returns cos and sin
of the pendulum angles to avoid discontinuity at 2π
"""
function measure(G::QubeServo2, xt::AbstractVector, ut=0, t=0; noisy=false, rng=Random.GLOBAL_RNG)
    y = xt[1:2]
    noisy && (y .+= G.σv*randn(rng, G.ny))
    return y
end
function measure(G::QubeServo2, xt::AbstractMatrix, ut=0, t=0; noisy=false, rng=Random.GLOBAL_RNG)
    y = xt[1:2,:]
    noisy && (y .+= G.σv*randn(rng, G.ny, size(xt,2)))
    return y
end

"""
    ct_dynamics(G::QubeServo2, xt, ut)

Compute continuous dynamics function f(x,u), where
ẋ = f(x,u) is the nonlinear state-space model.

Dynamics are computed about the reference point, voltage
is the input signal.
"""
function ct_dynamics(G::QubeServo2, xt, ut)

    # Voltage saturation, clamp numerical instability
    ut = clamp.(ut, -G.v_max, G.v_max)
    xt = clamp.(xt, -1e5, 1e5)

    # Extract variables
    if typeof(xt) <: AbstractMatrix
        α = xt[2:2,:]
        θd = xt[3:3,:]
        αd = xt[4:4,:]
    elseif typeof(xt) <: AbstractVector
        α = xt[2:2]
        θd = xt[3:3]
        αd = xt[4:4]
    end

    cosα = cos.(α)
    sinα = sin.(α)

    # Compute coefficients
    a1 = G.mp * G.Lr^2 .+ G.Jp * sinα.^2 .+ G.Jr 
    a2 = 0.5 * G.mp * G.Lp * G.Lr * cosα 
    a3 = 2 * G.Jp * sinα .* cosα .* θd .* αd - 
         0.5 * G.mp * G.Lp * G.Lr * sinα .* αd.^2 +
         (G.Dr + G.km^2 / G.Rm) * θd

    b1 = G.km/G.Rm

    c1 = 0.5 * G.mp * G.Lp * G.Lr * cosα
    c2 = G.Jp
    c3 = -G.Jp * cosα .* sinα .* θd.^2 +
         0.5 * G.mp * G.Lp * G.g * sinα + G.Dp*αd

    # Stick it in an equation
    denom = 1 ./ (a1 .* c2 - a2 .* c1)
    θdd = denom .* (c2 .* (b1 .* ut - a3) + a2 .* c3)
    αdd = denom .* (-c1 .* (b1 .* ut - a3) - a1 .* c3)

    # Return derivative
    return [θd; αd; θdd; αdd]

end

"""
    (G::QubeServo2)(dxt, ut, t=0; noisy=false, rng=...)

Compute discrete dynamics with the midpoint method, adding
white noise to the state output
"""
function (G::QubeServo2)(xt, ut, t=0; noisy=false, rng=Random.GLOBAL_RNG)

    # RK4 method
    k1 = ct_dynamics(G, xt, ut)
    k2 = ct_dynamics(G, xt + G.Δt*k1/2, ut)
    k3 = ct_dynamics(G, xt + G.Δt*k2/2, ut)
    k4 = ct_dynamics(G, xt + G.Δt*k3, ut)
    xt = xt + G.Δt*(k1 + 2k2 + 2k3 + k4)/6

    # Add noise
    noisy && (xt .+= G.σw*randn(rng, size(xt)...))

    return xt

end

"""
    init_state(G::QubeServo2, batches::Int; zero_init=false, rng=...)

Initialise system state with uniform probability between limits
"""
function init_state(G::QubeServo2, batches::Int; zero_init=false, rng=Random.GLOBAL_RNG)
    zero_init && (return zeros(G.nx,batches))
    x0 = 2*rand(rng, G.nx, batches) .- 1
    x0 .*= G.x0_lims
end

function init_state(G::QubeServo2; zero_init=false, rng=Random.GLOBAL_RNG)
    zero_init && (return zeros(G.nx))
    x0 = 2*rand(rng, G.nx) .- 1
    x0 .*= G.x0_lims
end


"""
    linearised_qube2(; Δt = 0.05, max_steps = 60, upwards=true)

Get linearised Qube Servo 2 environment
States are zero with pendulum arm in the upright equilibrium
"""
function linearised_qube2(;
    Δt = 0.02, 
    max_steps = 150,
    x0_lims = fill(1,4),
    upwards=true,
    rng = StableRNG(0)
)

    # Motor
    Rm = 8.4            # Resistance
    kt = 0.042          # Current-torque (N-m/A)
    km = 0.042          # Back-emf constant (V-s/rad)

    # Rotary Arm
    Mr = 0.095          # Mass (kg)
    Lr = 0.085          # Total length (m)
    Jr = Mr*Lr^2/12     # Moment of inertia about pivot (kg-m^2)
    Dr = 0.0015         # Equivalent Viscous Damping Coefficient (N-m-s/rad)
 
    # Pendulum Link
    mp = 0.024          # Mass (kg)
    Lp = 0.129          # Total length (m)
    Jp = mp*Lp^2/12     # Moment of inertia about pivot (kg-m^2)
    Dp = 0.0005         # Equivalent Viscous Damping Coefficient (N-m-s/rad)
    g = 9.81            # Gravity Constant

    # Parallel axis theorem
    Jr = Jr + 0.25*Mr*Lr^2
    Jp = Jp + 0.25*mp*Lp^2

    # Mass, damping, stiffness matrices
    upwards ? (s = 1) : (s = -1)
    M = zeros(2,2)
    V = diagm([Dr, Dp])
    K = zeros(2,2)
    R = reshape([1.0, 0], 2, 1)
    M[1,1] = mp*Lr^2 + Jr
    M[1,2] = -(s/2)*mp*Lp*Lr
    M[2,1] = M[1,2]
    M[2,2] = Jp
    K[2,2] = -(s/2)*mp*Lp*g

    # State-space form
    A = [zeros(2,2) Matrix(I(2)); -M\K -M\V]
    B = [zeros(2,1); M\R]
    C = Matrix([I(2) zeros(2,2)])

    # Add actuator dynamics (voltage input, not torque)
    A[3,3] = A[3,3] - kt*kt/Rm*B[3]
    A[4,3] = A[4,3] - kt*kt/Rm*B[4]
    B = kt * B / Rm

    # Convert to discrete-time system
    Ag = (I(4) + Δt*A)
    Bg = Δt * B
    Cg = C

    # Noise
    σw = 1e-2
    σv = 1e-2

    # Return lti system
    G = lti(
        Ag, Bg, Cg; 
        x0_lims=x0_lims, σw=σw, σv=σv, 
        max_steps=max_steps, rng=rng
    )

    return G

end