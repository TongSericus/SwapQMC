decode_basis(i::Int64, NsX::Int64) = [(i - 1) % NsX + 1, div(i - 1, NsX) + 1]
encode_basis(i::Int64, j::Int64, NsX::Int64) = i + (j - 1) * NsX

function one_body_matrix_ionic_hubbard_2D(NsX::Int64, NsY::Int64, t::Float64, δ::Float64)
    """
        Cartesian lattice coordinates:
        (1,3) (2,3) (3,3)       7 8 9
        (1,2) (2,2) (3,2)  ->   4 5 6
        (1,1) (2,1) (3,1)       1 2 3
    """

    Ns = NsX * NsY
    T = zeros(Ns, Ns)
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

function auxfield_matrix_hubbard(σ::AbstractArray{Int64}, auxfield::Matrix{Float64})
    """
        Hubbard HS field matrix generator
    """
    pfield = isone.(σ)
    mfield = isone.(-σ)
    
    V₊ = pfield * auxfield[1,1] .+ mfield * auxfield[2,1]
    V₋ = pfield * auxfield[1,2] .+ mfield * auxfield[2,2]

    return V₊, V₋
end

function singlestep_matrix!(
    B₊::AbstractMatrix{T}, B₋::AbstractMatrix{T}, 
    B₊ext::AbstractMatrix{T}, B₋ext::AbstractMatrix{T}, idx::Int64,
    σ::AbstractArray{Int64}, system::IonicHubbardExtended;
    useFirstOrderTrotter::Bool = system.useFirstOrderTrotter,
    tmpmat = similar(B₊)
) where {T<:Number}
    """
        Compute the propagator matrix and its extender
    """
    Bₖ = system.Bk
    BΔ = system.BΔ

    Aidx = system.Aidx
    Bidx = system.Bidx

    V₊, V₋ = auxfield_matrix_hubbard(σ, system.auxfield)
    @. V₊ *= BΔ
    @. V₋ *= BΔ

    if useFirstOrderTrotter
        mul!(B₊, Bₖ, Diagonal(V₊))
        mul!(B₋, Bₖ, Diagonal(V₋))

        @views B₊ext[Aidx, Aidx] .= B₊[Aidx, Aidx]
        @views B₊ext[Bidx.+ idx, Bidx.+ idx] .= B₊[Bidx, Bidx]
        @views B₊ext[Aidx, Bidx .+ idx] .= B₊[Aidx, Bidx]
        @views B₊ext[Bidx .+ idx, Aidx] .= B₊[Bidx, Aidx]

        @views B₋ext[Aidx, Aidx] .= B₋[Aidx, Aidx]
        @views B₋ext[Bidx .+ idx, Bidx .+ idx] .= B₋[Bidx, Bidx]
        @views B₋ext[Aidx, Bidx .+ idx] .= B₋[Aidx, Bidx]
        @views B₋ext[Bidx .+ idx, Aidx] .= B₋[Bidx, Aidx]
    else
        mul!(tmpmat, Diagonal(V₊), Bₖ)
        mul!(B₊, Bₖ, tmpmat)
        
        mul!(tmpmat, Diagonal(V₋), Bₖ)
        mul!(B₋, Bₖ, tmpmat)
    end

    return nothing
end
