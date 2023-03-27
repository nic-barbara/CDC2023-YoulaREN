"""
$(TYPEDEF)

Environment for 1D magnetic levitation with an
electromagnetic coil and a ball.

See Khalil, Nonlinear Systems (2002), exercises 1.18 and 12.8
"""
mutable struct MagLevEnv <: AbstractEnvironment
    nx::Int64
    nu::Int64
    ny::Int64
    max_steps::Int64
    x0_lims::AbstractVector
    rng::AbstractRNG
    σw                                  # Process noise covariance is σw.^2
    σv                                  # Measurement noise covariance is σv.^2
    m                                   # Mass of magnet (kg)
    k                                   # Viscous friction coefficient (N/m/s)
    g                                   # Gravitational acceleration (m/s^2)
    a                                   # A regularising length? (m)
    L0                                  # Inductive constant (H)
    L1                                  # Inductive constant (H)
    R                                   # Series resistance of circuit (Ω)
    Δt                                  # Sample time (s)
    z_min                               # Minimum vertical position (m)
    z_max                               # Maximum vertical position (m)
    v_min                               # Minimum voltage (V)
    v_max                               # Maximum voltage (V)
end

"""
    MagLevEnv(;...)

Construct magnetic levitation environment with sample time `Δt`,
maximum simulation steps `max_steps`, initial condition range
`x0_lims`, and random seed `rng`.
"""
function MagLevEnv(;
    Δt = 0.005,
    max_steps = 100,
    x0_lims = [0.03,0.1,0.0],
    rng = StableRNG(0)
)

    # Constants from Exercise 12.8 of Khalil (2002)
    m = 0.1
    k = 0.001
    g = 9.81
    a = 0.05
    L0 = 0.01
    L1 = 0.02
    R = 1

    # Min/max vertical positions, voltage
    z_min = 0
    z_max = 0.1
    v_min = 0
    v_max = 15

    # Noise
    σw = [5e-4, 5e-4, 5e-4]
    σv = [1e-3, 1e-3]

    # Sizes
    nx, nu, ny = 3, 1, 2

    # Construct model
    return MagLevEnv(
        nx, nu, ny, max_steps, x0_lims, rng, σw, σv,
        m, k, g, a, L0, L1, R, Δt, z_min, z_max,
        v_min, v_max
    )

end

"""
    measure(G::MagLevEnv, xt::AbstractVector, ...)

Compute measurement given current state.
"""
function measure(G::MagLevEnv, xt::AbstractVector, ut=0, t=0; noisy=false, rng=Random.GLOBAL_RNG)
    noisy ? (xt[[1,3]] .+ G.σv.*randn(rng, G.ny)) : (xt[[1,3]])
end
function measure(G::MagLevEnv, xt::AbstractMatrix, ut=0, t=0; noisy=false, rng=Random.GLOBAL_RNG)
    noisy ? (xt[[1,3],:] .+ G.σv.*randn(rng, G.ny, size(xt,2))) : (xt[[1,3],:])
end

"""
    ct_dynamics(G::MagLevEnv, xt, ut)

Compute continuous dynamics function f(x,u), where
ẋ = f(x,u) is the nonlinear state-space model.
"""
function ct_dynamics(G::MagLevEnv, xt, ut)
    
    # Voltage saturation, clamp numerical instability
    ut = clamp.(ut, G.v_min, G.v_max)
    xt = clamp.(xt, -1e6, 1e6)

    # Extract states (pos, vel, current)
    if typeof(xt) <: AbstractMatrix
        z = xt[1:1,:]
        zd = xt[2:2,:]
        ic = xt[3:3,:]
    elseif typeof(xt) <: AbstractVector
        z = xt[1:1]
        zd = xt[2:2]
        ic = xt[3:3]
    end

    # Intermediate calcs
    A1 = G.L0 * G.a * ic ./ (G.a .+ z).^2
    Lx1 = G.L1 .+ G.a * G.L0 ./ (G.a .+ z)

    # Compute derivatives
    zdd = G.g .- (G.k/G.m) .* zd .- A1 .* ic / 2G.m
    icd = (ut .- G.R .* ic .+ A1 .* zd) ./ Lx1

    # Return derivatives
    return [zd; zdd; icd]

end

"""
    (G::MagLevEnv)(xt, ut, t=0; noisy=false, rng=...)

Compute discrete dynamics with the midpoint method, adding
white noise to the state output
"""
function (G::MagLevEnv)(xt, ut, t=0; noisy=false, rng=Random.GLOBAL_RNG)
    
    # Midpoint method
    k1 = ct_dynamics(G, xt, ut)
    k2 = ct_dynamics(G, xt .+ G.Δt*k1/2, ut)
    xt = xt + G.Δt*k2 
    
    # Add noise
    noisy && (xt .+= G.σw.*randn(rng, size(xt)...))

    # Check for contact
    xt = apply_contact_conditions(G,xt)

    return xt
end

"""
    init_state(G::MagLevEnv, batches::Int; zero_init=false, rng=...)

Initialise system state with uniform probability between limits.
Current (A) randomly selected in positive range, centred around
equilibrium current required to hold the ball at y = 0m.
"""
function init_state(G::MagLevEnv, batches::Int; zero_init=false, rng=Random.GLOBAL_RNG)

    # Generate initial conditions
    zero_init && (return zeros(G.nx,batches))
    x0 = (2*rand(rng, G.nx, batches) .- 1) .* G.x0_lims

    # Add non-zero mean to position and current
    m1 = mean([G.z_min, G.z_max])
    m3 = current_equilibrium(G)
    x0 = x0 .+ [m1, 0, m3]

    # Check contact and return
    x0 = apply_contact_conditions(G,x0)
    return x0
end

function init_state(G::MagLevEnv; zero_init=false, rng=Random.GLOBAL_RNG)

    # Generate initial conditions
    zero_init && (return zeros(G.nx))
    x0 = (2*rand(rng, G.nx) .- 1) .* G.x0_lims

    # Add non-zero mean to position and current
    m1 = mean([G.z_min, G.z_max])
    m3 = current_equilibrium(G)
    x0 = x0 .+ [m1, 0, m3]

    # Check contact and return
    x0 = apply_contact_conditions(G,x0)
    return x0
end

# Current required to hold the ball at y = r m
current_equilibrium(G, r=0) = sqrt(2*G.m*G.g*(G.a + r)^2 / (G.L0*G.a))

"""
    apply_contact_conditions(G::MagLevEnv, xt::AbstractVecOrMat)

Saturate position and negate velocity at limits of environment
"""
function apply_contact_conditions(G::MagLevEnv, xt::AbstractVecOrMat)

    # Check for contact
    if typeof(xt) <: AbstractMatrix
        
        xtp = _contact_pos.(xt[1:1,:], G.z_max, G.z_min)
        xtv = _contact_vel.(xt[1:1,:], xt[2:2,:], G.z_max, G.z_min)
        xt = [xtp; xtv; xt[3:3,:]]
        
    elseif typeof(xt) <: AbstractVector
 
        if xt[1] > G.z_max
            xt = [G.z_max, -xt[2], xt[3]]
        elseif xt[1] < G.z_min
            xt = [G.z_min, -xt[2], xt[3]]
        end

    end

    return xt

end

# Helper to set position on contact
function _contact_pos(x, xmax, xmin) 
    if x > xmax
        return xmax
    elseif x < xmin
        return xmin
    else
        return x
    end
end

# Helper to set velocity on contact
_contact_vel(x, v, xmax, xmin) = ((x > xmax) || (x < xmin)) ? -v : v
