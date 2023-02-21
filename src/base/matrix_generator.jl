"""
    Generate Low-level Matrices Used in QMC
"""

# convert 2D coordinate indices into array indices
decode_basis(i::Int, NsX::Int) = [(i - 1) % NsX + 1, div(i - 1, NsX) + 1]
encode_basis(i::Int, j::Int, NsX::Int) = i + (j - 1) * NsX

function decode_basis_bilayer(i::Int, NsX::Int, NsY::Int)
    V = NsX * NsY

    # lower level
    i <= V && begin
        ix, iy = decode_basis(i, NsX)
        return ix, iy, -1
    end

    # upper level
    ix, iy = decode_basis(i - V, NsX)
    return ix, iy, 1
end

function encode_basis_bilayer(i::Int, j::Int, z::Int, NsX::Int, NsY::Int)
    # lower level
    z == -1 && begin
        return encode_basis(i, j, NsX)
    end

    V = NsX * NsY
    return encode_basis(i, j, NsX) + V
end

### Kinetic/One-body Matrix  ###
function one_body_matrix_bilayer_hubbard(NsX::Int, NsY::Int, t::Float64, t′::Float64)
    """
        Cartesian lattice coordinates:
        upper layer:
        (1,3) (2,3) (3,3)       16 17 18
        (1,2) (2,2) (3,2)  ->   13 14 15
        (1,1) (2,1) (3,1)       10 11 12
        lower layer:
        (1,3) (2,3) (3,3)       7  8  9
        (1,2) (2,2) (3,2)  ->   4  5  6
        (1,1) (2,1) (3,1)       1  2  3
    """
    Ns = NsX * NsY * 2
    T = zeros(Ns, Ns)

    for i = 1 : Ns
        ix, iy, iz = decode_basis_bilayer(i, NsX, NsY)
        # indices of nearest neighbours (nn) of (i, j)
        nn_up = mod(iy, NsY) + 1
        nn_dn = mod(iy - 2, NsY) + 1
        nn_lf = mod(ix, NsX) + 1
        nn_rg = mod(ix - 2, NsX) + 1

        # intra-layer hopping
        T[i, encode_basis_bilayer(ix, nn_up, iz, NsX, NsY)] = -t
        T[i, encode_basis_bilayer(ix, nn_dn, iz, NsX, NsY)] = -t
        T[i, encode_basis_bilayer(nn_lf, iy, iz, NsX, NsY)] = -t
        T[i, encode_basis_bilayer(nn_rg, iy, iz, NsX, NsY)] = -t

        # inter-layer hopping
        T[i, encode_basis_bilayer(ix, iy, -iz, NsX, NsY)] = -t′
    end

    return T
end

function one_body_matrix_ionic_hubbard_1D(Ns::Int, t::Float64, δ::Float64)
    T = zeros(Ns, Ns)

    # staggered potential is a 1D chain
    Δ = zeros(Ns)

    T = zeros(Ns, Ns)

    for i =  1 : Ns
        # indices of nearest neighbours (nn) of i
        nn_lf = mod(i, Ns) + 1
        nn_rg = mod(i - 2, Ns) + 1

        T[i, nn_lf] = -t
        T[i, nn_rg] = -t

        Δ[i] = (-1)^(i) * δ / 2
    end

    return T, Δ
end

function one_body_matrix_ionic_hubbard_2D(NsX::Int, NsY::Int, t::Float64, δ::Float64)
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
"""
    Hubbard HS field matrix generator
"""
function auxfield_matrix_hubbard(
    σ::AbstractArray{Int}, auxfield::Vector{T};
    V₊ = zeros(T, length(σ)),
    V₋ = zeros(T, length(σ)),
    isComplexHST::Bool = false
) where T
    isComplexHST && begin
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
"""
    Compute the propagator matrix for Bilayer Hubbard Model
"""
function imagtime_propagator!(
    B₊::AbstractMatrix{T}, B₋::AbstractMatrix{T},
    σ::AbstractArray{Int}, system::BilayerHubbard;
    useFirstOrderTrotter::Bool = system.useFirstOrderTrotter,
    tmpmat = similar(B₊)
) where {T<:Number}
    Bₖ = system.Bk

    auxfield_matrix_hubbard(
        σ, system.auxfield, 
        V₊=system.V₊, V₋=system.V₋,
        isComplexHST = system.useComplexHST
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
    σ::AbstractArray{Int}, system::BilayerHubbard;
    useFirstOrderTrotter::Bool = system.useFirstOrderTrotter,
    tmpmat = similar(B₊)
) where {T<:Number}
    Bₖ = system.Bk

    auxfield_matrix_hubbard(
        σ, system.auxfield, 
        V₊=system.V₊, V₋=system.V₋,
        isComplexHST = system.useComplexHST
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

"""
    Compute the propagator matrix for Ionic Hubbard Model
"""
function imagtime_propagator!(
    B₊::AbstractMatrix{T}, B₋::AbstractMatrix{T},
    σ::AbstractArray{Int}, system::IonicHubbard;
    useFirstOrderTrotter::Bool = system.useFirstOrderTrotter,
    tmpmat = similar(B₊)
) where {T<:Number}
    Bₖ = system.Bk
    BΔ = system.BΔ

    auxfield_matrix_hubbard(
        σ, system.auxfield, 
        V₊=system.V₊, V₋=system.V₋,
        isComplexHST = system.useComplexHST
    )
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
