"""
$(TYPEDEF)

A Youla model type to wrap around some dynamic model (eg: REN).
Internal state contains the state of both the observer and Q-param,
with notation as follows:
    - Main system:      `x1, y = dynamics(x, u)`
    - Observer model:   `x̂1, ŷ = observer(x̂, u)`
    - Youla model:      `x̄1, v = model(x̄, ỹ)`
where `ỹ = y - ŷ` is the model error. We store internal states together
in an array: `ξ = [x̂, xq]`.
"""
mutable struct YoulaModel <: AbstractFeedback
    Q
    G
    C
    nx::Int
    Filter
end

"""
    YoulaModel(Q, G, C; filterType=NoFilter) 

Initialise Youla model with Youla parameter Q, system model G,
and base controller C.

Option to include a batch-norm filter on the model inputs.
"""
function YoulaModel(Q, G, C; filterType=NoFilter) 
    filt = filterType()
    YoulaModel(Q, G, C, (G.nx + Q.nx), filt)
end

"""
    Flux.trainable(m::YoulaModel)

Define trainable parameters
"""
Flux.trainable(m::YoulaModel) = Flux.trainable(m.Q)

"""
    internal_states(ξ::AbstractVector, n::Int)

Split internal states into observer and Youla parameter states
"""
internal_states(ξ::AbstractVector, n::Int) = ξ[1:n], ξ[(n+1):end]
internal_states(ξ::AbstractMatrix, n::Int) = ξ[1:n, :], ξ[(n+1):end, :]

"""
    (m::YoulaModel)(ξ, y, u, t=0; rng=Random.GLOBAL_RNG)

Call Youla model given internal states, measurements, previous
control input (set to 0 if unimportant), and time.
"""
function (m::YoulaModel)(ξ, y, u, t=0; rng=Random.GLOBAL_RNG)

    # Get states and innovations
    x̂, x̄ = internal_states(ξ, m.G.nx)
    ỹ = y - measure(m.G, x̂, u, t; rng=rng)

    # Evaluate Youla parameter and controls
    x̄, v = m.Q(x̄, m.Filter(ỹ), t)
    u = control_action(m.C, x̂, y, t) + v

    # Update state and store
    x̂ = observer_update(m.C, x̂, y, u, t; rng=rng)
    ξ1 = vcat(x̂, x̄)

    return ξ1, u
    
end

"""
$(TYPEDEF)

Define a wrapper to go with some dynamic model (eg: REN).
Puts model in direct feedback with an output-feedback controller.
Internal state contains the state of both the observer and model,
with notation as follows:
    - Main system:      x1, y = dynamics(x, u)
    - Observer model:   x̂1, ŷ = observer(x̂, u)
    - NN model:         x̄1, v = model(x̄, ỹ)
where ỹ = y - ŷ is the model error. We store internal states together
in an array: ξ = [x̂, xq].
"""
mutable struct FeedbackModel <: AbstractFeedback
    Q
    G
    C
    nx::Int
    Filter
end

"""
    FeedbackModel(Q, G, C; filterType=NoFilter) 

Initialise Feedback model with augmenting parameter Q, 
system model G, and base controller C.

Option to include a batch-norm filter on the model inputs.
"""
function FeedbackModel(Q, G, C; filterType=NoFilter) 
    filt = filterType()
    FeedbackModel(Q, G, C, (G.nx + Q.nx), filt)
end

"""
    Flux.trainable(m::FeedbackModel)

Define trainable parameters
"""
Flux.trainable(m::FeedbackModel) = Flux.trainable(m.Q)

"""
    (m::FeedbackModel)(ξ, y, u, t=0; rng=Random.GLOBAL_RNG)

Call Feedback model given internal states, measurements, previous
control input (set to 0 if unimportant), and time.
"""
function (m::FeedbackModel)(ξ, y, u, t=0; rng=Random.GLOBAL_RNG)

    # Get states
    x̂, x̄ = internal_states(ξ, m.G.nx)

    # Evaluate feedback augmentation and controls
    x̄, v = m.Q(x̄, m.Filter(y), t)
    u = control_action(m.C, x̂, y, t) + v

    # Update state and store
    x̂ = observer_update(m.C, x̂, y, u, t; rng=rng)
    ξ1 = vcat(x̂, x̄)

    return ξ1, u
    
end


"""
$(TYPEDEF)

Youla model with a separate observer to the base controller.
Documentation is similar to the YoulaModel
"""
mutable struct YoulaModelObsv <: AbstractFeedback
    Q
    G
    C
    obsv
    nx::Int
    Filter
end

"""
"""
function YoulaModelObsv(Q, G, C, obsv; filterType=NoFilter) 
    filt = filterType()
    YoulaModelObsv(Q, G, C, obsv, (G.nx + Q.nx + obsv.nx), filt)
end

"""
"""
Flux.trainable(m::YoulaModelObsv) = Flux.trainable(m.Q)

"""
"""
function internal_states(m::YoulaModelObsv, ξ::AbstractVector)
    n1 = m.C.nx
    n2 = m.obsv.nx
    return ξ[1:n1], ξ[(n1+1):(n1+n2)], ξ[(n1+n2+1):end]
end
function internal_states(m::YoulaModelObsv, ξ::AbstractMatrix)
    n1 = m.C.nx
    n2 = m.obsv.nx
    return ξ[1:n1,:], ξ[(n1+1):(n1+n2),:], ξ[(n1+n2+1):end,:]
end

"""
"""
function (m::YoulaModelObsv)(ξ, y, u, t=0; rng=Random.GLOBAL_RNG)

    # Get states and innovations
    x̂, x̂h, x̄ = internal_states(m, ξ)
    ỹh = y - measure(m.G, x̂h, u, t; rng=rng)

    # Evaluate Youla parameter and controls
    x̄, v = m.Q(x̄, m.Filter(ỹh), t)
    u = control_action(m.C, x̂, y, t) + v

    # Update state and store
    x̂ = observer_update(m.C, x̂, y, u, t; rng=rng)
    x̂h = observer_update(m.obsv, x̂h, y, u, t; rng=rng)
    ξ1 = vcat(x̂, x̂h, x̄)

    return ξ1, u
    
end
