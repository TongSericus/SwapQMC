"""
    Generate Low-level Matrices Used in QMC
"""

# convert 2D coordinate indices into array indices
decode_basis(i::Int, NsX::Int) = [(i - 1) % NsX + 1, div(i - 1, NsX) + 1]
encode_basis(i::Int, j::Int, NsX::Int) = i + (j - 1) * NsX

### Auxiliary-field Matrix ###
function auxfield_matrix_hubbard(
    σ::AbstractArray{Int}, auxfield::Vector{T};
    V₊ = zeros(T, length(σ)),
    V₋ = zeros(T, length(σ)),
    isChargeHST::Bool = false
) where T
    """
        Hubbard HS field matrix generator
    """
    isChargeHST && begin
        for i in eachindex(σ)
            isone(σ[i]) ? (idx₊ = idx₋ = 1) : (idx₊ = idx₋ = 2)
            V₊[i] = auxfield[idx₊]
            V₋[i] = auxfield[idx₋]
        end
        
        return V₊, V₋
    end
    
    for i in eachindex(σ)
        isone(σ[i]) ? (idx₊ = 1; idx₋ = 2) : (idx₊ = 2; idx₋ = 1)
        V₊[i] = auxfield[idx₊]
        V₋[i] = auxfield[idx₋]
    end
    
    return V₊, V₋
end

### Combined Matrix ###
function imagtime_propagator!(
    B₊::AbstractMatrix{T}, B₋::AbstractMatrix{T},
    σ::AbstractArray{Int}, system::GenericHubbard;
    useFirstOrderTrotter::Bool = system.useFirstOrderTrotter,
    tmpmat = similar(B₊)
) where {T<:Number}
    """
        Compute the propagator matrix for generic Hubbard Model 
        (spin decomposition is used, up and down parts are different)
    """
    Bₖ = system.Bk

    auxfield_matrix_hubbard(
        σ, system.auxfield, 
        V₊=system.V₊, V₋=system.V₋,
        isChargeHST = system.useChargeHST
    )
    V₊, V₋ = system.V₊, system.V₋

    if useFirstOrderTrotter
        mul!(B₊, Bₖ, Diagonal(V₊))
        mul!(B₋, Bₖ, Diagonal(V₋))
    else
        mul!(tmpmat, Diagonal(V₊), Bₖ)
        mul!(B₊, Bₖ, tmpmat)
        
        mul!(tmpmat, Diagonal(V₋), Bₖ)
        mul!(B₋, Bₖ, tmpmat)
    end

    return nothing
end

function imagtime_propagator!(
    B::AbstractMatrix{T},
    σ::AbstractArray{Int}, system::GenericHubbard;
    useFirstOrderTrotter::Bool = system.useFirstOrderTrotter,
    tmpmat = similar(B₊)
) where {T<:Number}
    """
        Compute the propagator matrix for generic Hubbard Model
        (charge decomposition is used, up and down parts are the same)
    """
    Bₖ = system.Bk

    auxfield_matrix_hubbard(
        σ, system.auxfield, 
        V₊=system.V₊, V₋=system.V₋,
        isChargeHST = system.useChargeHST
    )
    V₊ = system.V₊

    if useFirstOrderTrotter
        mul!(B, Bₖ, Diagonal(V₊))
    else
        mul!(tmpmat, Diagonal(V₊), Bₖ)
        mul!(B, Bₖ, tmpmat)
    end

    return nothing
end
