"""
    get_model(G, Cb, model_type, hp, model_init=nothing, 
              ftype=NoFilter; rng=Random.GLOBAL_RNG)

Get a feedback control model specified by `model_type` given a plant `G`, a
base controller `Cb`, and a hyperparameter dictionary `hp`.

The `model_type` argument can contain any (sensible) combination of the
following options:

- Feedback architecture: `youla` or `feedback`
- Neural network: `lstm`, `ren`, `gren`
- Initialiser for ren: `longinit`, `init` (uses `model_init` for a contracting ren)
- Add an extra observer for Youla: `obsv`

Each option should be separated by a hyphen. For example,`youla-ren-longinit`.

Option to include something to initialise the model in the `model_init` 
argument. Specify and input filter for feedback architectures with `ftype`. 
Default is none.
"""
function get_model(
    G, Cb, hp, model_init=nothing; 
    rng=Random.GLOBAL_RNG
)

    ftype = hp["filtertype"]
    model_type = hp["modeltype"]
    _check(s) = occursin(s, model_type)

    ########### Base model ###########

    if _check("lstm")

        nv    = hp["nv"]
        m     = lstm(G.ny, nv, G.nu; rng=rng)

    elseif _check("gren")

        γ     = hp["γ"]
        nx    = hp["nx"]
        nv    = hp["nv"]
        ϕfunc = hp["ϕfunc"]

        # Fix D22 = 0 for now
        ps = LipschitzRENParams{Float64}(
            G.ny, nx, nv, G.nu, γ; 
            nl=ϕfunc, polar_param=true, rng=rng, D22_zero=true, 
        )
        _check("small") && small_ren!(ps, hp; rng=rng)
        m = WrapREN(ps)

    elseif _check("ren")

        nx    = hp["nx"]
        nv    = hp["nv"]
        ϕfunc = hp["ϕfunc"]

        # Longinit initialises a slower dynamic model
        if _check("longinit")

            # Current initialisation occasionally produces an error
            good_init = false
            while !good_init
                try 
                    ps = ContractingRENParams{Float64}(
                        G.ny, nx, nv, G.nu; nl=ϕfunc, 
                        polar_param=true, rng=rng, init=:cholesky
                    )
                    good_init = true
                catch
                end
            end
            _check("small") && small_ren!(ps, hp; rng=rng)

        elseif _check("init")

            A, B, C, D = model_init.A, model_init.B, model_init.C, model_init.D
            ps = ContractingRENParams(nv, A, B, C, D; nl=ϕfunc, 
                                      polar_param=true, rng=rng)
            _check("small") && small_ren!(ps, hp; rng=rng)

        else
            ps = ContractingRENParams{Float64}(G.ny, nx, nv, G.nu; nl=ϕfunc, 
                                               polar_param=true, rng=rng)
            _check("small") && small_ren!(ps, hp; rng=rng)
        end
        m = WrapREN(ps)

    else

        error("Unrecognsie model type $model_type")
        
    end

    ########### Feedback architecture ###########

    if _check("feedback")

        set_output_zero!(m)
        model = FeedbackModel(m, G, Cb; filterType=ftype)

    elseif _check("youla")

        set_output_zero!(m)

        if _check("obsv")
            obsv = model_init
            model = YoulaModelObsv(m, G, Cb, obsv; filterType=ftype)
        else
            model = YoulaModel(m, G, Cb; filterType=ftype)
        end

    else

        model = m
        
    end

    return model

end

"""
    small_ren!(ps::AbstractRENParams, hp::Dict; rng=Random.GLOBAL_RNG)

Helper function to reduce dimensionality of REN models
"""
function small_ren!(ps::AbstractRENParams, hp::Dict; rng=Random.GLOBAL_RNG)
    div_fact = hp["div_factor"]
    n1 = 2*ps.nx + ps.nv
    n2 = round(Int, n1/div_fact)
    ps.direct.X = ps.direct.X[1:n2,:]
end
