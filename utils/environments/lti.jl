"""
$(TYPEDEF)

Linear Time Invariant (LTI) system environment structure,
including typical disturbance variances and simulation horizon
"""
mutable struct lti <: LTI
    A
    B
    C
    D
    σw::AbstractFloat               # (To add Guassian white process noise)
    σv::AbstractFloat               # (To add Guassian white measurement noise)
    nx::Int64
    nu::Int64
    ny::Int64
    max_steps::Int64
    x0_lims::AbstractVector         # Standard deviations to sample initial state (default ±1)
    rng::AbstractRNG
end

"""
    lit(A, B, C, D=0; ...)

Construct LTI system from state-space matrices.

Options to specify noise covariances, simulation time horizon
with `max_steps`, and random seed.
"""
function lti(
    A, B, C, D = 0;
    σw         = 0.0, 
    σv         = 0.0,
    max_steps  = 200, 
    x0_lims    = [], 
    rng        = Random.GLOBAL_RNG
)
    nx, nu, ny = size(A,1), size(B, 2), size(C,1)
    isempty(x0_lims) && (x0_lims = ones(typeof(A[1,1]), nx))
    (D == 0) && (D = zeros(ny,nu))
    lti(A, B, C, D, σw, σv, nx, nu, ny, max_steps, x0_lims, rng)
end

"""
    (G::LTI)(xt, ut, t=0; noisy=false, rng=...)

Compute next state of an LTI system given current state and input. 
If `noisy = true` is specified, adds Gaussian random noise with
stdev `G.σw` to state.
"""
function (G::LTI)(xt, ut, t=0; noisy=false, rng=Random.GLOBAL_RNG)
    xt1 = G.A * xt + G.B * ut
    noisy && (xt1 .+=  G.σw*randn(rng, size(xt)...))
    return xt1
end

"""
    measure(G::LTI, xt, ut, t=0; noisy=false, rng=...)

Compute output of LTI system given current state and input. If 
`noisy = true` is specified, adds Gaussian random noise with stdev
`G.σv` to output.
"""
function measure(G::LTI, xt, ut, t=0; noisy=false, rng=Random.GLOBAL_RNG)
    yt = G.C * xt + G.D * ut
    noisy && (yt .+= G.σv*randn(rng, size(yt)...))
    return yt 
end

"""
    init_state(G::LTI, batches::Int; zero_init=false, rng=...)

Initialise system state either as zeros or normally-
distributed random numbers.
"""
function init_state(G::LTI, batches::Int; zero_init=false, rng=Random.GLOBAL_RNG)
    zero_init && (return zeros(G.nx,batches))
    return G.x0_lims .* randn(rng, G.nx, batches)
end

function init_state(G::LTI; zero_init=false, rng=Random.GLOBAL_RNG)
    zero_init && (return zeros(G.nx))
    return G.x0_lims .* randn(rng, G.nx)
end
