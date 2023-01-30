"""
    Generate Low-level Matrices Used in QMC
"""

# convert 2D coordinate indices into array indices
decode_basis(i::Int64, NsX::Int64) = [(i - 1) % NsX + 1, div(i - 1, NsX) + 1]
encode_basis(i::Int64, j::Int64, NsX::Int64) = i + (j - 1) * NsX

function decode_basis_bilayer(i::Int64, NsX::Int64)
    if isodd(i)
        ix, iy = decode_basis(div(i+1, 2), NsX)
        return ix, iy, -1
    else
        ix, iy = decode_basis(div(i, 2), NsX)
        return ix, iy, 1
    end
end

function encode_basis_bilayer(i::Int64, j::Int64, z::Int64, NsX::Int64)
    if z == -1
        return 2*(i + (j - 1) * NsX) - 1
    else
        return 2*(i + (j - 1) * NsX)
    end
end

### Kinetic/One-body Matrix  ###
function one_body_matrix_bilayer_hubbard(NsX::Int64, NsY::Int64, t::Float64, t′::Float64)
    """
        Cartesian lattice coordinates:
        upper layer:
        (1,3) (2,3) (3,3)       14 16 18
        (1,2) (2,2) (3,2)  ->   8  10 12
        (1,1) (2,1) (3,1)       2  4  6
        lower layer:
        (1,3) (2,3) (3,3)       13 15 17
        (1,2) (2,2) (3,2)  ->   7  9  11
        (1,1) (2,1) (3,1)       1  3  5
    """
    Ns = NsX * NsY * 2
    T = zeros(Ns, Ns)

    for i = 1 : Ns
        ix, iy, iz = decode_basis_bilayer(i, NsX)
        # indices of nearest neighbours (nn) of (i, j)
        nn_up = mod(iy, NsY) + 1
        nn_dn = mod(iy - 2, NsY) + 1
        nn_lf = mod(ix, NsX) + 1
        nn_rg = mod(ix - 2, NsX) + 1

        # intra-layer hopping
        T[i, encode_basis_bilayer(ix, nn_up, iz, NsX)] = -t
        T[i, encode_basis_bilayer(ix, nn_dn, iz, NsX)] = -t
        T[i, encode_basis_bilayer(nn_lf, iy, iz, NsX)] = -t
        T[i, encode_basis_bilayer(nn_rg, iy, iz, NsX)] = -t

        # inter-layer hopping
        T[i, encode_basis_bilayer(ix, iy, -iz, NsX)] = -t′
    end

    return T
end

function one_body_matrix_ionic_hubbard_1D()
end

function one_body_matrix_ionic_hubbard_2D(NsX::Int64, NsY::Int64, t::Float64, δ::Float64)
    """
        Cartesian lattice coordinates:
        (1,3) (2,3) (3,3)       7 8 9
        (1,2) (2,2) (3,2)  ->   4 5 6
        (1,1) (2,1) (3,1)       1 2 3
    """

    Ns = NsX * NsY
    T = zeros(Ns, Ns)

    # staggered potential is a 1D chain
    Δ = zeros(Ns)

    for i = 1 : Ns
        ix, iy = decode_basis(i, NsX)
        # indices of nearest neighbours (nn) of (i, j)
        nn_up = mod(iy, NsY) + 1
        nn_dn = mod(iy - 2, NsY) + 1
        nn_lf = mod(ix, NsX) + 1
        nn_rg = mod(ix - 2, NsX) + 1

        T[i, encode_basis(ix, nn_up, NsX)] = -t
        T[i, encode_basis(ix, nn_dn, NsX)] = -t
        T[i, encode_basis(nn_lf, iy, NsX)] = -t
        T[i, encode_basis(nn_rg, iy, NsX)] = -t

        Δ[i] = (-1)^(ix + iy) * δ / 2
    end

    return T, Δ
end

### Auxiliary-field Matrix ###
function auxfield_matrix_hubbard(
    σ::AbstractArray{Int64}, auxfield::Vector{T};
    V₊ = zeros(T, length(σ)),
    V₋ = zeros(T, length(σ)),
    isComplexHST::Bool = false
) where T
    """
        Hubbard HS field matrix generator
    """
    if isComplexHST
        for i in eachindex(σ)
            isone(σ[i]) ? (idx₊ = 1; idx₋ = 1) : (idx₊ = 2; idx₋ = 2)
            V₊[i] = auxfield[idx₊]
            V₋[i] = auxfield[idx₋]
        end
    else
        for i in eachindex(σ)
            isone(σ[i]) ? (idx₊ = 1; idx₋ = 2) : (idx₊ = 2; idx₋ = 1)
            V₊[i] = auxfield[idx₊]
            V₋[i] = auxfield[idx₋]
        end
    end
    
    return V₊, V₋
end

### Combined Matrix ###
function singlestep_matrix!(
    B₊::AbstractMatrix{T}, B₋::AbstractMatrix{T},
    σ::AbstractArray{Int64}, system::BilayerHubbard;
    useFirstOrderTrotter::Bool = system.useFirstOrderTrotter,
    tmpmat = similar(B₊)
) where {T<:Number}
    """
        Compute the propagator matrix for Bilayer Hubbard Model
    """
    Bₖ = system.Bk

    auxfield_matrix_hubbard(σ, system.auxfield, V₊=system.V₊, V₋=system.V₋)
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

function singlestep_matrix!(
    B₊::AbstractMatrix{T}, B₋::AbstractMatrix{T},
    σ::AbstractArray{Int64}, system::IonicHubbard;
    useFirstOrderTrotter::Bool = system.useFirstOrderTrotter,
    tmpmat = similar(B₊)
) where {T<:Number}
    """
        Compute the propagator matrix for Ionic Hubbard Model
    """
    Bₖ = system.Bk
    BΔ = system.BΔ

    auxfield_matrix_hubbard(σ, system.auxfield, V₊=system.V₊, V₋=system.V₋)
    V₊, V₋ = system.V₊, system.V₋
    @. V₊ *= BΔ
    @. V₋ *= BΔ

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
