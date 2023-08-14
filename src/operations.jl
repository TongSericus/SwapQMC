###################################
##### Basic operations in QMC #####
###################################
"""
    compute_Metropolis_ratio(G, α, i, sidx)

    Compute the ratio of determinants using Green's function
    det(M_new) / det(M_old) = 1 + α * (1 - G[i, i])
    if only the ith auxiliary field is flipped
"""
function compute_Metropolis_ratio(
    G::Vector{T}, α::Ta, i::Int, sidx::Int
) where {T<:AbstractMatrix, Ta}
    d_up = 1 + α[1, i] * (1 - G[1][sidx, sidx])
    d_dn = 1 + α[2, i] * (1 - G[2][sidx, sidx])
    r = abs(d_up * d_dn)

    return r, d_up, d_dn
end

"""
    compute_Metropolis_ratio(G, α, i, sidx)

    Compute the ratio of determinants using Green's function, with
    the auxiliary field being complex and spin-up and spin-down channel
    are identical (hence only a single G is required)
"""
function compute_Metropolis_ratio(
    G::AbstractMatrix, α::Ta, sidx::Int;
    forceSymmetry::Bool = false
) where {Ta<:Number}
    d = α * (1 - G[sidx, sidx])
    # accept ratio
    r = isreal(α) ? (1 + d)^2 : (1 + d)^2 / (α + 1)
    forceSymmetry && (r = (1 + d) * conj(1 + d))

    return r, d+1
end

function Base.kron!(C::AbstractMatrix, α::Number, a::AbstractVector, b::AbstractVector)
    """
        kron!(C, α, a, b)

        In-place operation of C = α * u * wᵀ
    """
    for i in eachindex(a)
        for j in eachindex(a)
            @inbounds C[j, i] = α * a[j] * b[i]
        end
    end
end

"""
    update_G!(G, α, d, sidx, ws)

    Fast update the Green's function at the ith site:
    G ← G - α_{i, σ} / d_{i, i} * u * wᵀ
    where u = (I - G)e₁, w = Gᵀe₁
"""
function update_G!(
    G::AbstractMatrix{T}, 
    α::Ta, d::Td, sidx::Int64, 
    ws::LDRWorkspace{T, E}; direction::Int = 1
) where {T, Ta, Td, E}
    ImG = ws.M
    dG = ws.M′

    direction == 1 && begin
        g = @view G[sidx, :]
        Img = @view ImG[:, sidx]

        # compute I - G
        for i in eachindex(g)
            @inbounds Img[i] = -G[i, sidx]
        end
        Img[sidx] += 1

        # compute (I - G)eᵢ * Gᵀeᵢ
        @views kron!(dG, α / d, Img, g)

        return G .-= dG
    end

    g = @view G[:, sidx]
    Img = @view ImG[sidx, :]

    # compute I - G
    for i in eachindex(g)
        @inbounds Img[i] = -G[sidx, i]
    end
    Img[sidx] += 1

    # compute Geᵢ * (I-G)ᵀeᵢ
    @views kron!(dG, α / d, g, Img)

    return G .-= dG
end

"""
    wrap_G!(G, B, ws)

    Compute the wrapped Green's function that would be
    used for the propagation in the next time slice, i.e.,
        G ← B * G * B⁻¹ (direction = 1), or
        G ← B⁻¹ * G * B (direction = 2)
"""
function wrap_G!(
    G::AbstractMatrix{T}, B::AbstractMatrix{Tb}, ws::LDRWorkspace{T, E};
    direction::Int = 1
) where {T, Tb, E}    

    B⁻¹ = ws.M′
    copyto!(B⁻¹, B)
    inv_lu!(B⁻¹, ws.lu_ws)
    
    direction == 1 ? (
                        mul!(ws.M, B, G);
                        mul!(G, ws.M, B⁻¹)
                    ) : 
                    (
                        mul!(ws.M, B⁻¹, G);
                        mul!(G, ws.M, B)
                    )
    return G
end

"""
    wrap_G!(G, B, B⁻¹, ws)

    Compute the wrapped Green's function, and B⁻¹ is given.
    Note: B and B⁻¹ are not necessarily matrices but can be checkerboard decompositions.
"""
function wrap_G!(
    G::AbstractMatrix{T}, B::Tb, B⁻¹::Tb, 
    ws::LDRWorkspace{T, E}
) where {T, Tb, E}

    mul!(ws.M, B, G);
    mul!(G, ws.M, B⁻¹)

    return G
end

flip_HSField(σ::Int) = Int64((σ + 1) / 2 + 1)

"""
    prod_cluster!(B::AbstractMatrix, Bl::AbstractArray{T}, C::AbstractMatrix)

    In-place calculation of prod(Bl) and overwrite the result to B, with an auxiliary matrix C
"""
function prod_cluster!(B::AbstractMatrix, Bl::AbstractArray{T}, C::AbstractMatrix) where {T<:AbstractMatrix}
    size(B) == size(Bl[1]) == size(C) || throw(BoundsError())
    k = length(Bl)
    k == 1 && (copyto!(B, Bl[1]); return nothing)
    k == 2 && (mul!(B, Bl[1], Bl[2]); return nothing)

    mul!(C, Bl[1], Bl[2])
    @inbounds for i in 3:k
        mul!(B, C, Bl[i])
        copyto!(C, B)
    end

    return nothing
end

############################################
##### Full Imaginary-time Propagations #####
############################################
"""
    build_propagator(auxfield, system, qmc, ws)

    Propagate over the full space-time lattice given the auxiliary field configuration
"""
function build_propagator(
    auxfield::AbstractMatrix{Int64}, system::System, qmc::QMC, ws::LDRWorkspace{T,E}; 
    isReverse::Bool = true, K = qmc.K, K_interval = qmc.K_interval
) where {T, E}
    V = system.V
    si = qmc.stab_interval

    # initialize partial matrix products
    Tb = eltype(system.auxfield)
    B = [Matrix{Tb}(I, V, V), Matrix{Tb}(I, V, V)]
    MatProd = Cluster(V, 2 * K, T = Tb)
    F = ldrs(B[1], 2)
    FC = Cluster(B = ldrs(B[1], 2 * K))

    Bm = MatProd.B
    Bf = FC.B

    isReverse && begin
        for i in K:-1:1
            for j = 1 : K_interval[i]
            @views σ = auxfield[:, (i - 1) * si + j]
            imagtime_propagator!(B[1], B[2], σ, system, tmpmat = ws.M)
            Bm[i] = B[1] * Bm[i]            # spin-up
            Bm[K + i] = B[2] * Bm[K + i]    # spin-down
        end

        # save all partial products
        copyto!(Bf[i], F[1])
        copyto!(Bf[K + i], F[2])

        rmul!(F[1], Bm[i], ws)
        rmul!(F[2], Bm[K + i], ws)
    end

        return F, MatProd, FC
    end

    for i in 1:K
        for j = 1 : K_interval[i]
            @views σ = auxfield[:, (i - 1) * si + j]
            imagtime_propagator!(B[1], B[2], σ, system, tmpmat = ws.M)
            Bm[i] = B[1] * Bm[i]            # spin-up
            Bm[K + i] = B[2] * Bm[K + i]    # spin-down
        end

        copyto!(Bf[i], F[1])
        copyto!(Bf[K + i], F[2])

        lmul!(Bm[i], F[1], ws)
        lmul!(Bm[K + i], F[2], ws)
    end

    return F, MatProd, FC
end

"""
    build_propagator!(Fc, MatProd, ws)

    Propagate over the full space-time lattice given the matrix clusters
"""
function build_propagator!(
    Fc::Vector{Fact}, MatProd::Cluster{C}, ws::LDRWorkspace{T,E};
    K = div(length(MatProd.B), 2),
    isReverse::Bool = true, isSymmetric::Bool = false
) where {Fact, C, T, E}
    V = size(MatProd.B[1])
    i = eltype(ws.M) <: Real ? 1.0 : 1.0+0.0im
    F = ldrs(Matrix(i*I, V), 2)

    Bm = MatProd.B

    isReverse && begin 
        for n in K:-1:1
            copyto!(Fc[n], F[1])
            isSymmetric || copyto!(Fc[K + n], F[2])

            rmul!(F[1], Bm[n], ws)
            isSymmetric || rmul!(F[2], Bm[K + n], ws)
        end

        return F
    end

    for n in 1:K
        copyto!(Fc[n], F[1])
        isSymmetric || copyto!(Fc[K + n], F[2])

        lmul!(Bm[n], F[1], ws)
        isSymmetric || lmul!(Bm[K + n], F[2], ws)
    end
    
    return F
end

###############################################
##### Imaginary-time-displaced Operations #####
###############################################

"""
    compute_Metropolis_ratio(system, replica, walker, α, sidx, ridx)

    Compute the Metroplis accept ratio for the ridx-th replica at the sidx-th site,
    using the formula
    r↑ = r↓ = 1 + α * (1 - Gτ[sidx, sidx] - Γ).

    The overall ratio needs a phase factor for complex HS transform
    r = r↑ * r↓ / (α + 1)
"""
function compute_Metropolis_ratio(
    system::System,
    replica::Replica{W, T}, walker::W,
    α::Ta, sidx::Int, ridx::Int;
    direction::Int = 1, forceSymmetry::Bool = false
) where {W, T, Ta}

    # set alias
    Aidx = replica.Aidx
    GA⁻¹ = replica.GA⁻¹
    a = replica.a
    b = replica.b
    λₖ = replica.λₖ
    Gτ = walker.G[1][sidx, sidx]

    # direction=2 -> back propagation
    direction == 1 ? (
            Bk = system.Bk; 
            Bk⁻¹ = system.Bk⁻¹;
            G0τ = walker.G0τ[1];
            Gτ0 = walker.Gτ0[1]
        ) : 
        (
            Bk = system.Bk⁻¹; 
            Bk⁻¹ = system.Bk;
            G0τ = walker.Gτ0[1];
            Gτ0 = walker.G0τ[1]
        )
    
    # compute Γ = a * bᵀ
    if system.useFirstOrderTrotter  # asymmetric case
        ridx == 1 ? 
            begin
                @views mul!(a, GA⁻¹, G0τ[Aidx, sidx])
                @views transpose_mul!(b, Gτ0[sidx, Aidx], replica.Im2GA)
            end :
            # update the second replica
            begin
                t = replica.t
                @views mul!(t, replica.Im2GA, G0τ[Aidx, sidx])
                @views mul!(a, GA⁻¹, t)
                @views copyto!(b, Gτ0[sidx, Aidx])
            end
    else                            # symmetric case
        # update the first replica
        ridx == 1 ? 
            begin
                BG0τ = replica.t
                @views mul!(BG0τ, Bk⁻¹[Aidx, :], G0τ[:, sidx])
                @views mul!(a, GA⁻¹, BG0τ)

                Gτ0B = replica.t
                @views transpose_mul!(Gτ0B, Gτ0[sidx, :], Bk[:, Aidx])
                @views transpose_mul!(b, Gτ0B, replica.Im2GA)
            end : 
        # update the second replica
            begin
                t = replica.t
                @views mul!(a, Bk⁻¹[Aidx, :], G0τ[:, sidx])
                @views mul!(t, replica.Im2GA, a)
                @views mul!(a, GA⁻¹, t)

                @views transpose_mul!(b, Gτ0[sidx, :], Bk[:, Aidx])
            end
    end
    Γ = transpose(a) * b
    
    # regular DQMC ratio
    d = 1 + α * (1 - Gτ)
    # ratio of detgA (Grover matrix) with a thermaldynamic integration variable (λₖ)
    dᵧ = (1 - α*Γ/d)^λₖ
    ## accept ratio
    r = isreal(α) ? (d*dᵧ)^2 : (d*dᵧ)^2 / (α+1)
    forceSymmetry && (r = (d*dᵧ) * conj(d*dᵧ))

    γ = α / d
    ρ = α / (d - α * Γ)

    return r, γ, ρ
end

function update_G0!(
    G0::AbstractMatrix{T}, γ::Ta, 
    Gτ0::AbstractMatrix{T}, G0τ::AbstractMatrix{T},
    sidx::Int64, ws::LDRWorkspace{T, E}
) where {T, Ta, E}
    dG0 = ws.M

    # compute (I - G)eᵢ * Gᵀeᵢ
    @views kron!(dG0, γ, G0τ[:, sidx], Gτ0[sidx, :])

    @. G0 += dG0
end

"""
    update_Gτ0!(Gτ0, γ, Gτ, sidx, ws)

    Update the imaginary time Green's function G(τ,0) when the
    sidx-th spin is flipped
"""
function update_Gτ0!(
    Gτ0::AbstractMatrix{T}, γ::Ta, 
    Gτ::AbstractMatrix{T},
    sidx::Int64, ws::LDRWorkspace{T, E};
    direction::Int = 1
) where {T, Ta, E}
    GτmI = ws.M
    dGτ0 = ws.M′

    direction == 1 && begin
        g = @view Gτ[sidx, :]
        gτmI = @view GτmI[:, sidx]

        # compute I - G
        for i in eachindex(g)
            @inbounds gτmI[i] = Gτ[i, sidx]
        end
        gτmI[sidx] -= 1

        # compute (I - G)eᵢ * Gᵀeᵢ
        @views kron!(dGτ0, γ, gτmI, Gτ0[sidx, :])

        return @. Gτ0 += dGτ0
    end

    g = @view Gτ[:, sidx]
    gτmI = @view GτmI[sidx, :]

    # compute I - G
    for i in eachindex(g)
        @inbounds gτmI[i] = Gτ[sidx, i]
    end
    gτmI[sidx] -= 1

    # compute (I - G)eᵢ * Gᵀeᵢ
    @views kron!(dGτ0, γ, Gτ0[:, sidx], gτmI)

    return @. Gτ0 += dGτ0
end

"""
    update_G0τ!(G0τ, γ, Gτ, sidx, ws)

    Update the imaginary time Green's function G(0,τ) when the
    sidx-th spin is flipped
"""
function update_G0τ!(
    G0τ::AbstractMatrix{T}, γ::Ta, 
    Gτ::AbstractMatrix{T},
    sidx::Int64, ws::LDRWorkspace{T, E};
    direction::Int = 1
) where {T, Ta, E}
    dG0τ = ws.M

    direction == 1 && begin
        # compute (I - G)eᵢ * Gᵀeᵢ
        @views kron!(dG0τ, γ, G0τ[:, sidx], Gτ[sidx, :])

        return @. G0τ += dG0τ
    end

    # compute (I - G)eᵢ * Gᵀeᵢ
    @views kron!(dG0τ, γ, Gτ[:, sidx], G0τ[sidx, :])

    return @. G0τ += dG0τ
end

"""
    update_invGA!(replica, ρ)

    Update the Grover inverse 
    GA⁻¹ = (GA₁ * GA₂ + (I-GA₁) * (I-GA₂))⁻¹
    when the sidx-th spin is flipped
"""
function update_invGA!(replica::Replica{W, T}, ρ::Tp) where {W, T, Tp}
    GA⁻¹ = replica.GA⁻¹
    a = replica.a
    b = replica.b
    bᵀ = replica.t
    dGA⁻¹ = replica.ws.M

    @views transpose_mul!(bᵀ, b, GA⁻¹)

    kron!(dGA⁻¹, ρ, a, bᵀ)
    @. GA⁻¹ += dGA⁻¹
end

"""
    wrap_Gs!(Gτ, Gτ0, G0τ, B, ws)

    Wrap Green's function at time τ and two time-displaced Green's function Gτ0 and G0τ
    to the next time slice as
    
        G(τ+Δτ) <- B(Δτ)G(τ)B⁻¹(Δτ)
        G(τ+Δτ, 0) <- B(Δτ)G(τ, 0)
        G(0, τ+Δτ) <- G(0, τ)B⁻¹(Δτ)
"""
function wrap_Gs!(
    Gτ::AbstractMatrix{T}, Gτ0::AbstractMatrix{T}, G0τ::AbstractMatrix{T},
    B::Tb, ws::LDRWorkspace{T, E}; direction::Int = 1
) where {T, Tb, E}
    # compute B⁻¹
    B⁻¹ = ws.M′
    copyto!(B⁻¹, B)
    inv_lu!(B⁻¹, ws.lu_ws)
    
    direction == 1 && begin
        # update G(τ)
        mul!(ws.M, B, Gτ)
        mul!(Gτ, ws.M, B⁻¹)

        # update G(τ,0)
        mul!(ws.M, B, Gτ0)
        copyto!(Gτ0, ws.M)

        # update G(0,τ)
        mul!(ws.M, G0τ, B⁻¹)
        copyto!(G0τ, ws.M)

        return nothing
    end

    # update G(τ)
    mul!(ws.M, B⁻¹, Gτ)
    mul!(Gτ, ws.M, B)

    # update G(τ,0)
    mul!(ws.M, Gτ0, B)
    copyto!(Gτ0, ws.M)

    # update G(0,τ)
    mul!(ws.M, B⁻¹, G0τ)
    copyto!(G0τ, ws.M)
    
    return nothing
end
