"""
$(TYPEDEF)

Dummy filter returns unmodified input
"""
mutable struct NoFilter <: AbstractFilter
end

"""
    update!(filt::NoFilter)

Dummy filter has no parameters to update
"""
update!(filt::NoFilter) = nothing

"""
    (filt::NoFilter)(x)

Return unchanged input
"""
(filt::NoFilter)(x) = x
