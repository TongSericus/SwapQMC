"""
    Swap Monte Carlo Propagation in the GC
"""
### Basic operations ###
function compute_Metropolis_ratio(
    G::AbstractArray{T}, α::Ta, i::Int, sidx::Int
) where {T, Ta}
    d_up = 1 + α[1, i] * (1 - G[1][sidx, sidx])
    d_dn = 1 + α[2, i] * (1 - G[2][sidx, sidx])
    r = abs(d_up * d_dn)

    return r, d_up, d_dn
end

"""
    Compute the ratio of determinants using Green's function, with
    the auxiliary field being complex and spin-up and spin-down channel
    are identical (hence only a single G is required)
"""
function compute_Metropolis_ratio(
    G::AbstractMatrix, α::Ta, i::Int, sidx::Int
) where Ta
    γ = α[1, i]
    d = γ * (1 - G[sidx, sidx])
    # accept ratio
    r = (1 + d)^2 / (γ + 1)

    return real(r), d+1
end

"""
    kron!(C, α, a, b)

    In-place operation of C = α * u * wᵀ
"""
function Base.kron!(C::AbstractMatrix, α::Number, a::AbstractVector, b::AbstractVector)
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
function update_G!(G::AbstractMatrix{T}, α::Ta, d::Ta, sidx::Int64, ws::LDRWorkspace{T, E}) where {T, Ta, E}
    ImG = ws.M
    dG = ws.M′
    g = @view G[sidx, :]
    Img = @view ImG[:, sidx]

    # compute I - G
    for i in eachindex(g)
        @inbounds Img[i] = -G[i, sidx]
    end
    Img[sidx] += 1

    # compute (I - G)eᵢ * Gᵀeᵢ
    @views kron!(dG, α / d, Img, g)

    G .-= dG
end

"""
    wrap_G!(G, B, ws)

    Compute G ← B * G * B⁻¹
"""
function wrap_G!(G::AbstractMatrix{T}, B::AbstractMatrix{TB}, ws::LDRWorkspace{T, E}) where {T, TB, E}
    mul!(ws.M, B, G)
    
    B⁻¹ = ws.M′
    copyto!(B⁻¹, B)
    inv_lu!(B⁻¹, ws.lu_ws)
    mul!(G, ws.M, B⁻¹)
end

function wrap_G!(G::AbstractMatrix{T}, B::AbstractMatrix{TB}, B⁻¹::AbstractMatrix{TB}, ws::LDRWorkspace{T, E}) where {T, TB, E}
    mul!(ws.M, B, G)
    mul!(G, ws.M, B⁻¹)
end

flip_HSField(σ::Int) = Int64((σ + 1) / 2 + 1)

###################################
### Sweep in the Z_{A, 2} space ###
###################################
function update_cluster!(
    walker::HubbardGCWalker{Float64, Ft, E, C}, 
    swapper::HubbardGCSwapper{Float64, E, Ft},
    extsys::ExtendedSystem{T}, qmc::QMC, cidx::Int, ridx::Int
) where {T<:Hubbard, Ft, E, C}
    system = extsys.system
    LA = extsys.LA
    LB = extsys.LB

    k = qmc.K_interval[cidx]
    K = qmc.K

    G = swapper.G
    B⁺ = swapper.B
    Bk = swapper.Bk[ridx]
    Bk⁻¹ = swapper.Bk⁻¹[ridx]
    ws = swapper.ws

    Bl = walker.Bl.B
    cluster = walker.Bc
    α = walker.α

    for i in 1 : k
        l = (cidx - 1) * qmc.stab_interval + i
        @views σ = flip_HSField.(walker.auxfield[:, l])

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || (wrap_G!(G[1], Bk, Bk⁻¹, ws); wrap_G!(G[2], Bk, Bk⁻¹, ws))

        for j in 1 : system.V
            sidx = j
            ridx == 2 && j > LA && (sidx = j - LA + system.V)

            # compute ratios of determinants through G
            r, d_up, d_dn = compute_Metropolis_ratio(G, α, σ[j], sidx)

            if rand() < r
                # accept the move, update the field and the Green's function
                walker.auxfield[j, l] *= -1
                update_G!(G[1], α[1, σ[j]], d_up, sidx, ws)
                update_G!(G[2], α[2, σ[j]], d_dn, sidx, ws)
            end
        end

        # compute G <- Bk⁻¹ * G * Bk to restore the ordering
        system.useFirstOrderTrotter || (wrap_G!(G[1], Bk⁻¹, Bk, ws); wrap_G!(G[2], Bk⁻¹, Bk, ws))
        
        @views σ = walker.auxfield[:, l]
        imagtime_propagator!(Bl[i], Bl[k + i], σ, system, tmpmat = walker.ws.M)

        # rank-1 update of the Green's function
        expand!(B⁺, Bl[i], LA, LB, ridx)
        wrap_G!(G[1], B⁺, ws)
        expand!(B⁺, Bl[k + i], LA, LB, ridx)
        wrap_G!(G[2], B⁺, ws)
    end

    @views copyto!(cluster.B[cidx], prod(Bl[k:-1:1]))
    @views copyto!(cluster.B[K + cidx], prod(Bl[2*k:-1:k+1]))

    return nothing
end

"""
    sweep!(extsys::ExtendedSystem, qmc::QMC, s::Swapper, w::Walker, ridx::Int)

    Sweep the walker over the extended Hilbert space over the imaginary time from 0 to β (ridx=1) or from β to 2β (ridx=2)
"""
function sweep!(
    extsys::ExtendedSystem{T}, qmc::QMC, 
    swapper::HubbardGCSwapper{Float64, E, Ft}, 
    walker::HubbardGCWalker{Float64, Ft, E, C}, ridx::Int
) where {T<:Hubbard, Ft, E, C}
    K = qmc.K
    
    ws = swapper.ws
    F = swapper.F
    # temporal factorizations
    C⁺ = swapper.C
    L⁺ = swapper.L
    R⁺ = swapper.R

    Bc = walker.Bc.B
    tmpL = walker.FC.B
    tmpR = walker.Fτ

    for cidx in 1 : K
        update_cluster!(walker, swapper, extsys, qmc, cidx, ridx)

        # multiply the updated slice to the right factorization
        lmul!(Bc[cidx], tmpR[1], walker.ws)
        # then expand to the larger factorization
        expand!(R⁺, tmpR[1], ridx)
        expand!(L⁺, tmpL[cidx], ridx)
        # then merge the right, central and left factorizations,
        # note that B¹_{cidx} is at the leftmost position, i.e.,
        # U = B¹_{cidx-1}⋯B¹_{1} * U² * B¹_{L}⋯B¹_{cidx+1}B¹_{cidx}
        mul!(F[1], R⁺, C⁺[1], ws)
        rmul!(F[1], L⁺, ws)

        # same step for spin-dn
        lmul!(Bc[K + cidx], tmpR[2], walker.ws)
        expand!(R⁺, tmpR[2], ridx)
        expand!(L⁺, tmpL[K + cidx], ridx)
        mul!(F[2], R⁺, C⁺[2], ws)
        rmul!(F[2], L⁺, ws)
        
        # G needs to be periodically recomputed
        update!(swapper)
    end

    # At the end of the simulation, recompute all partial factorizations
    run_full_propagation(walker.Bc, walker.ws, FC = walker.FC)

    # save Fτs
    copyto!.(walker.F, tmpR)
    expand!(C⁺[1], walker.F[1], ridx)
    expand!(C⁺[2], walker.F[2], ridx)
    # then reset Fτs to unit matrices
    ldr!(tmpR[1], I)
    ldr!(tmpR[2], I)

    return nothing
end

#############################
### Sweep in the Z² space ###
#############################
function update_cluster!(
    walker::HubbardGCWalker{Float64, Ft, E, C},
    system::Hubbard, qmc::QMC, cidx::Int
) where {Ft, E, C}
    k = qmc.K_interval[cidx]
    K = qmc.K

    Bk = system.Bk
    Bk⁻¹ = system.Bk⁻¹

    G = walker.G
    ws = walker.ws
    Bl = walker.Bl.B
    cluster = walker.Bc
    α = walker.α

    for i in 1 : k
        l = (cidx - 1) * qmc.stab_interval + i
        @views σ = flip_HSField.(walker.auxfield[:, l])

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || (wrap_G!(G[1], Bk, Bk⁻¹, ws); wrap_G!(G[2], Bk, Bk⁻¹, ws))

        for j in 1 : system.V
            # compute ratios of determinants through G
            r, d_up, d_dn = compute_Metropolis_ratio(G, α, σ[j], j)

            if rand() < r
                # accept the move, update the field and the Green's function
                walker.auxfield[j, l] *= -1
                update_G!(G[1], α[1, σ[j]], d_up, j, ws)
                update_G!(G[2], α[2, σ[j]], d_dn, j, ws)
            end
        end
        
        # compute G <- Bk⁻¹ * G * Bk to restore the ordering
        system.useFirstOrderTrotter || (wrap_G!(G[1], Bk⁻¹, Bk, ws); wrap_G!(G[2], Bk⁻¹, Bk, ws))

        @views σ = walker.auxfield[:, l]
        imagtime_propagator!(Bl[i], Bl[k + i], σ, system, tmpmat = ws.M)

        # rank-1 update of the Green's function
        wrap_G!(G[1], Bl[i], ws)
        wrap_G!(G[2], Bl[k + i], ws)
    end

    @views copyto!(cluster.B[cidx], prod(Bl[k:-1:1]))
    @views copyto!(cluster.B[K + cidx], prod(Bl[2*k:-1:k+1]))

    return nothing
end

"""
    sweep!(sys::System, qmc::QMC, s::Walker)

    Sweep a single walker over the imaginary time from 0 to β
"""
function sweep!(
    system::Hubbard, qmc::QMC, 
    walker::HubbardGCWalker{Float64, Ft, E, C}
) where {Ft, E, C}
    K = qmc.K

    ws = walker.ws
    Bc = walker.Bc.B
    tmpL = walker.FC.B
    tmpR = walker.Fτ
    tmpM = walker.F

    for cidx in 1 : K
        update_cluster!(walker, system, qmc, cidx)

        # multiply the updated slice to the right factorization
        lmul!(Bc[cidx], tmpR[1], ws)
        lmul!(Bc[K + cidx], tmpR[2], ws)
        # then merge the right and left factorizations,
        # note that B_{cidx} is at the leftmost position, i.e.,
        # U = B_{cidx-1}⋯B_{2}B_{1}⋯B_{cidx+1}B_{cidx}
        mul!(tmpM[1], tmpR[1], tmpL[cidx], ws)
        mul!(tmpM[2], tmpR[2], tmpL[K + cidx], ws)

        # G needs to be periodically recomputed from scratch
        update!(walker)
    end

    # At the end of the simulation, recompute all partial factorizations
    run_full_propagation(walker.Bc, walker.ws, FC = walker.FC)

    # save Fτs
    copyto!.(walker.F, tmpR)
    # then reset Fτs to unit matrices
    ldr!(tmpR[1], I)
    ldr!(tmpR[2], I)

    return nothing
end

################################################################################
# Symmetric Sweep for Complex HS Transform
################################################################################

###################################
### Sweep in the Z_{A, 2} space ###
###################################
"""
    The cluster is sweeped by assuming that the spin-up and
    spin-down channel is identical
"""
function update_cluster!(
    walker::HubbardGCWalker{ComplexF64, Ft, E, C}, swapper::HubbardGCSwapper{ComplexF64, E, Ft},
    extsys::ExtendedSystem{T}, qmc::QMC, cidx::Int, ridx::Int
) where {T<:Hubbard, Ft, E, C}
    system = extsys.system
    LA = extsys.LA
    LB = extsys.LB

    k = qmc.K_interval[cidx]

    G = swapper.G[1]    # only a single G is need
    B⁺ = swapper.B
    Bk = swapper.Bk[ridx]
    Bk⁻¹ = swapper.Bk⁻¹[ridx]
    ws = swapper.ws

    Bl = walker.Bl.B
    cluster = walker.Bc
    α = walker.α

    for i in 1 : k
        l = (cidx - 1) * qmc.stab_interval + i
        @views σ = flip_HSField.(walker.auxfield[:, l])

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || wrap_G!(G, Bk, Bk⁻¹, ws)

        for j in 1 : system.V
            sidx = j
            ridx == 2 && j > LA && (sidx = j - LA + system.V)

            # compute ratios of determinants through G
            r, d = compute_Metropolis_ratio(G, α, σ[j], sidx)

            if rand() < r
                # accept the move, update the field and the Green's function
                walker.auxfield[j, l] *= -1
                update_G!(G, α[1, σ[j]], d, sidx, ws)
            end
        end

        # compute G <- Bk⁻¹ * G * Bk to restore the ordering
        system.useFirstOrderTrotter || wrap_G!(G, Bk⁻¹, Bk, ws)
        
        @views σ = walker.auxfield[:, l]
        imagtime_propagator!(Bl[i], σ, system, tmpmat = walker.ws.M)

        # rank-1 update of the Green's function
        expand!(B⁺, Bl[i], LA, LB, ridx)
        wrap_G!(G, B⁺, ws)
    end

    @views copyto!(cluster.B[cidx], prod(Bl[k:-1:1]))

    return nothing
end

function sweep!(
    extsys::ExtendedSystem{T}, qmc::QMC, 
    swapper::HubbardGCSwapper{ComplexF64, E, Ft}, 
    walker::HubbardGCWalker{ComplexF64, Ft, E, C}, ridx::Int
) where {T<:Hubbard, Ft, E, C}
    K = qmc.K
    
    ws = swapper.ws
    F = swapper.F
    # temporal factorizations
    C⁺ = swapper.C
    L⁺ = swapper.L
    R⁺ = swapper.R

    Bc = walker.Bc.B
    tmpL = walker.FC.B
    tmpR = walker.Fτ

    for cidx in 1 : K
        update_cluster!(walker, swapper, extsys, qmc, cidx, ridx)

        # multiply the updated slice to the right factorization
        lmul!(Bc[cidx], tmpR[1], walker.ws)
        # then expand to the larger factorization
        expand!(R⁺, tmpR[1], ridx)
        expand!(L⁺, tmpL[cidx], ridx)
        # then merge the right, central and left factorizations,
        # note that B¹_{cidx} is at the leftmost position, i.e.,
        # U = B¹_{cidx-1}⋯B¹_{1} * U² * B¹_{L}⋯B¹_{cidx+1}B¹_{cidx}
        mul!(F[1], R⁺, C⁺[1], ws)
        rmul!(F[1], L⁺, ws)
        
        # G needs to be periodically recomputed
        update!(swapper, identicalSpin=true)
    end

    # At the end of the simulation, recompute all partial factorizations
    run_full_propagation_oneside(walker.Bc, walker.ws, FC = walker.FC)

    # save Fτs
    copyto!(walker.F[1], tmpR[1])
    expand!(C⁺[1], walker.F[1], ridx)
    # then reset Fτs to unit matrices
    ldr!(tmpR[1], I)

    return nothing
end

#############################
### Sweep in the Z² space ###
#############################
function update_cluster!(
    walker::HubbardGCWalker{ComplexF64, Ft, E, C},
    system::Hubbard, qmc::QMC, cidx::Int
) where {Ft, E, C}
    k = qmc.K_interval[cidx]
    K = qmc.K

    Bk = system.Bk
    Bk⁻¹ = system.Bk⁻¹

    G = walker.G[1]
    ws = walker.ws
    Bl = walker.Bl.B
    cluster = walker.Bc
    α = walker.α

    for i in 1 : k
        l = (cidx - 1) * qmc.stab_interval + i
        @views σ = flip_HSField.(walker.auxfield[:, l])

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || wrap_G!(G, Bk, Bk⁻¹, ws)

        for j in 1 : system.V
            # compute ratios of determinants through G
            r, d = compute_Metropolis_ratio(G, α, σ[j], j)

            if rand() < r
                # accept the move, update the field and the Green's function
                walker.auxfield[j, l] *= -1
                update_G!(G, α[1, σ[j]], d, j, ws)
            end
        end
        
        # compute G <- Bk⁻¹ * G * Bk to restore the ordering
        system.useFirstOrderTrotter || wrap_G!(G, Bk⁻¹, Bk, ws)

        @views σ = walker.auxfield[:, l]
        imagtime_propagator!(Bl[i], σ, system, tmpmat = ws.M)

        # rank-1 update of the Green's function
        wrap_G!(G, Bl[i], ws)
    end

    @views copyto!(cluster.B[cidx], prod(Bl[k:-1:1]))

    return nothing
end

"""
    sweep!(sys::System, qmc::QMC, s::Walker)

    Sweep a single walker over the imaginary time from 0 to β
"""
function sweep!(
    system::Hubbard, qmc::QMC, 
    walker::HubbardGCWalker{ComplexF64, Ft, E, C}
) where {Ft, E, C}
    K = qmc.K

    ws = walker.ws
    Bc = walker.Bc.B
    tmpL = walker.FC.B
    tmpR = walker.Fτ
    tmpM = walker.F

    for cidx in 1 : K
        update_cluster!(walker, system, qmc, cidx)

        # multiply the updated slice to the right factorization
        lmul!(Bc[cidx], tmpR[1], ws)
        # then merge the right and left factorizations,
        # note that B_{cidx} is at the leftmost position, i.e.,
        # U = B_{cidx-1}⋯B_{2}B_{1}⋯B_{cidx+1}B_{cidx}
        mul!(tmpM[1], tmpR[1], tmpL[cidx], ws)

        # G needs to be periodically recomputed from scratch
        update!(walker, identicalSpin=true)
    end

    # At the end of the simulation, recompute all partial factorizations
    run_full_propagation_oneside(walker.Bc, walker.ws, FC = walker.FC)

    # save Fτs
    copyto!(walker.F[1], tmpR[1])
    # then reset Fτs to unit matrices
    ldr!(tmpR[1], I)

    return nothing
end
