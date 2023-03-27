"""
    init_state(m::AbstractCtrlModel; rng=nothing)

Return initial states full of zeros
"""
init_state(m::AbstractCtrlModel; rng=nothing) = zeros(m.nx)
init_state(m::AbstractCtrlModel, batches; rng=nothing) = zeros(m.nx, batches)

"""
    policy_reset!(m::AbstractFeedback)

Update the augmenting parameter if/when required.
"""
function policy_reset!(m::AbstractFeedback)
    policy_reset!(m.Q)
    update!(m.Filter)
    return nothing
end

"""
Additional functionality for REN wrapper in RobustNeuralNetworks.jl
"""
(m::WrapREN)(xt, ut, t; rng=nothing) = m(xt, ut)
(m::WrapREN)(ξ, y, u, t; rng=nothing) = m(ξ, y, t)
init_state(m::WrapREN; rng=nothing) = init_states(m)
init_state(m::WrapREN, batches; rng=nothing) = init_states(m, batches)
policy_reset!(m::WrapREN) = update_explicit!(m)

"""
    glorot_normal(n::Int, m::Int; T=Float64, rng=Random.GLOBAL_RNG)
Generate matrices or vectors from Glorot normal distribution
"""
glorot_normal(n::Int, m::Int; T=Float64, rng=Random.GLOBAL_RNG) = 
    convert.(T, randn(rng, n, m) / sqrt(n + m))
glorot_normal(n::Int; T=Float64, rng=Random.GLOBAL_RNG) = 
    convert.(T, randn(rng, n) / sqrt(n))
    
"""
    glorot_uniform(n::Int, m::Int; T=Float64, rng=Random.GLOBAL_RNG)
Generate matrices or vectors from Glorot uniform distribution
"""
glorot_uniform(n::Int, m::Int; T=Float64, rng=Random.GLOBAL_RNG) = 
    convert.(T, (rand(rng, n, m) .- 0.5) .* 12 / sqrt(n + m))
glorot_uniform(n::Int; T=Float64, rng=Random.GLOBAL_RNG) = 
    convert.(T, (rand(rng, n) .- 0.5) .* 12 / sqrt(n))
    