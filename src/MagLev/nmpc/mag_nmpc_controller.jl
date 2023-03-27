"""
$(TYPEDEF)

Nonlinear MPC controller for the MagLev system.
Uses high-gain observer for state estimation
"""
mutable struct MagLevNMPC
    G::MagLevEnv
    N::Int
    yref::Number
end
MagLevNMPC(G::MagLevEnv, N::Int) = MagLevNMPC(G, N, 0.05)

function (c::MagLevNMPC)(x::AbstractVector; verbose=false)

    # Extract useful constants
    N = c.N + 1
    Δt = c.G.Δt
    
    m = c.G.m
    k = c.G.k
    g = c.G.g
    a = c.G.a
    L0 = c.G.L0
    L1 = c.G.L1
    R = c.G.R

    # Min/max vertical positions, voltage
    z_min = c.G.z_min
    z_max = c.G.z_max
    v_min = c.G.v_min
    v_max = c.G.v_max

    # Create the JuMP model
    model = Model(Ipopt.Optimizer)
    verbose || set_silent(model)

    # Optimisation variables
    @variables(
        model, 
        begin
            z_min ≤ x1[1:N] ≤ z_max
            x2[1:N]
            x3[1:N]
            v_min ≤ u[1:N] ≤ v_max
        end
    )

    # Fix iniial conditions
    fix(x1[1], x[1]; force = true)
    fix(x2[1], x[2]; force = true)
    fix(x3[1], x[3]; force = true)

    # Define some expressions for convenience
    @NLexpressions(
        model,
        begin
            
            # Inductance
            Lx[j = 1:N], L1 + a*L0 / (a + x1[j])

            # Forcing term for x2_dot
            x2_dot[j = 1:N], g - (k/m) * x2[j] - (a * L0 / (2m)) * x3[j]^2 / (a + x1[j])^2

            # Internal forcing terms for x3_dot
            x3_dot1[j = 1:N], (-R * x3[j] + (a * L0) * x2[j] * x3[j] / (a + x1[j])^2) / Lx[j]

        end
    )

    # Build up dynamics with rectanguar integration
    for t in 1:N-1

        @NLconstraint(model, x1[t+1] == x1[t] + Δt * x2[t])

        @NLconstraint(
            model,
            x2[t+1] == x2[t] + Δt/2 * (x2_dot[t] + x2_dot[t+1])
        )

        @NLconstraint(
            model,
            x3[t+1] == x3[t] + Δt/2 * (
                x3_dot1[t] + u[t] / Lx[t] + 
                x3_dot1[t+1] + u[t+1] / Lx[t+1]  
            )
        )

    end
    
    # Objective function is near-quadratic
    @NLobjective(
        model, Min, 
        sum((1/q1^2)*(x1[t] - yref)^2 + (1/r1^2)*(u[t] - iref)^2 for t in 1:N-1)
    )

    # Solve the problem
    verbose && println("Solving...")
    optimize!(model)
    verbose && solution_summary(model)

    # Return the first control value
    return value.(u[1:1])

end

# So I can use this with MagLevCtrl
function MagLevCtrl(G::MagLevEnv, N::Int; ϵ=G.Δt)
    nx = 3
    L = reshape([1/ϵ, 1/ϵ^2], nx-1, 1)
    return MagLevCtrl(
        nx, L,
        MagLevNMPC(G, N)
    )
end