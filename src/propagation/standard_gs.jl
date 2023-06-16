"""
    Projective Monte Carlo Propagation in the regular Z space
"""

###################################################
##### Symmetric Sweep for Charge HS Transform #####
###################################################

function update_cluster!(
    walker::HubbardWalker{ComplexF64, Ft, E, C},
    system::Hubbard, qmc::QMC, cidx::Int
) where {Ft, E, C}
    k = qmc.K_interval[cidx]

    Bk = system.Bk
    Bk⁻¹ = system.Bk⁻¹

    G = walker.G[1]
    ws = walker.ws
    Bl = walker.Bl.B
    cluster = walker.Bc
    α = walker.α

    for i in 1 : k
        l = (cidx - 1) * qmc.stab_interval + i
        @views σ = walker.auxfield[:, l]

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || wrap_G!(G, Bk, Bk⁻¹, ws)

        for j in 1 : system.V
            σj = flip_HSField(σ[j])
            # compute ratios of determinants through G
            r, d = compute_Metropolis_ratio(G, α[1, σj], j)
            qmc.saveRatio && push!(walker.tmp_r, r)
            # accept ratio
            u = qmc.useHeatbath ? real(r) / (1 + real(r)) : real(r)

            if rand() < u
                # accept the move, update the field and the Green's function
                walker.auxfield[j, l] *= -1
                update_G!(G, α[1, σj], d, j, ws)
            end
        end
        
        # compute G <- Bk⁻¹ * G * Bk to restore the ordering
        system.useFirstOrderTrotter || wrap_G!(G, Bk⁻¹, Bk, ws)

        @views σ = walker.auxfield[:, l]
        imagtime_propagator!(Bl[i], σ, system, tmpmat = ws.M)

        # rank-1 update of the Green's function
        wrap_G!(G, Bl[i], ws)
    end

    @views prod_cluster!(cluster.B[cidx], Bl[k:-1:1], ws.M)

    return nothing
end

function sweep!(
    system::Hubbard, qmc::QMC, 
    walker::HubbardGCWalker{ComplexF64, Ft, E, C}
) where {Ft, E, C}
    """
        sweep!(system, qmc, walker)

        Sweep a single walker over the imaginary time from 0 to β
    """
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
    build_propagator(walker.Bc, walker.ws, FC = walker.FC, singleSided=true)

    # save Fτs
    copyto!(walker.F[1], tmpR[1])
    # then reset Fτs to unit matrices
    ldr!(tmpR[1], I)

    return nothing
end
