"""
    init_ctrl(G::AbstractEnvironment)

Helper function to initialise an array of zero control
inputs of the correct size.
"""
init_ctrl(G::AbstractEnvironment) = zeros(G.nu)
init_ctrl(G::AbstractEnvironment, batches) = zeros(G.nu, batches)