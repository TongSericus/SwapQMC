"""
    Swap Monte Carlo Propagation in the Z_{A, 2} space
"""

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
        @views σ = walker.auxfield[:, l]

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || (wrap_G!(G[1], Bk, Bk⁻¹, ws); wrap_G!(G[2], Bk, Bk⁻¹, ws))

        for j in 1 : system.V
            sidx = j
            ridx == 2 && j > LA && (sidx = j - LA + system.V)

            σj = flip_HSField(σ[j])
            # compute ratios of determinants through G
            r, d_up, d_dn = compute_Metropolis_ratio(G, α, σj, sidx)
            qmc.saveRatio && push!(walker.tmp_r, r)
            u = qmc.useHeatbath ? real(r) / (1 + real(r)) : real(r)

            if rand() < u
                # accept the move, update the field and the Green's function
                walker.auxfield[j, l] *= -1
                update_G!(G[1], α[1, σj], d_up, sidx, ws)
                update_G!(G[2], α[2, σj], d_dn, sidx, ws)
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

    @views prod_cluster!(cluster.B[cidx], Bl[k:-1:1], walker.ws.M)
    @views prod_cluster!(cluster.B[K + cidx], Bl[2*k:-1:k+1], walker.ws.M)

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
    build_propagator(walker.Bc, walker.ws, FC = walker.FC)

    # save Fτs
    copyto!.(walker.F, tmpR)
    expand!(C⁺[1], walker.F[1], ridx)
    expand!(C⁺[2], walker.F[2], ridx)
    # then reset Fτs to unit matrices
    ldr!(tmpR[1], I)
    ldr!(tmpR[2], I)

    return nothing
end

################################################################################
# Symmetric Sweep for Complex HS Transform
################################################################################
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
        @views σ = walker.auxfield[:, l]

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || wrap_G!(G, Bk, Bk⁻¹, ws)

        for j in 1 : system.V
            sidx = j
            ridx == 2 && j > LA && (sidx = j - LA + system.V)

            σj = flip_HSField(σ[j])
            # compute ratios of determinants through G
            r, d = compute_Metropolis_ratio(G, α[1, σj], sidx)
            qmc.saveRatio && push!(walker.tmp_r, r)
            u = qmc.useHeatbath ? real(r) / (1 + real(r)) : real(r)

            if rand() < u
                # accept the move, update the field and the Green's function
                walker.auxfield[j, l] *= -1
                update_G!(G, α[1, σj], d, sidx, ws)
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

    @views prod_cluster!(cluster.B[cidx], Bl[k:-1:1], walker.ws.M)

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
    build_propagator(walker.Bc, walker.ws, FC = walker.FC, singleSided=true)

    # save Fτs
    copyto!(walker.F[1], tmpR[1])
    expand!(C⁺[1], walker.F[1], ridx)
    # then reset Fτs to unit matrices
    ldr!(tmpR[1], I)

    return nothing
end
