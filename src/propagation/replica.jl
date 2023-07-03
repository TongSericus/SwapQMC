"""
    A new replica Monte Carlo sweep in the Z_{A, 2} space but has the same
    scaling as in the regular Z space, see the supplement of arXiv:2211.04334
    for full details
"""

function update_cluster!(
    walker::W, replica::Replica{W, ComplexF64, Float64},
    system::Hubbard, qmc::QMC, cidx::Int, ridx::Int
) where {W<:GCWalker}
    # set alias
    k = qmc.K_interval[cidx]
    Bk = system.Bk
    Bk⁻¹ = system.Bk⁻¹
    Gτ = walker.G[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    ws = walker.ws
    Bl = walker.Bl.B
    cluster = walker.Bc
    α = walker.α

    for i in 1 : k
        l = (cidx - 1) * qmc.stab_interval + i
        @views σ = walker.auxfield[:, l]

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || begin 
                                        wrap_G!(Gτ, Bk, Bk⁻¹, ws)
                                        wrap_G!(Gτ0, Bk, Bk⁻¹, ws)
                                        wrap_G!(G0τ, Bk, Bk⁻¹, ws)
                                    end

        for j in 1 : system.V
            σj = flip_HSField(σ[j])
            # compute ratios of determinants through G
            r, γ, ρ = compute_Metropolis_ratio(system, replica, walker, α[1, σj], j, ridx)
            qmc.saveRatio && push!(walker.tmp_r, r)
            # accept ratio
            u = qmc.useHeatbath ? real(r) / (1 + real(r)) : real(r)

            if rand() < u
                # accept the move, update the field and the Green's function
                walker.auxfield[j, l] *= -1
                
                ### rank-1 updates ###
                # update imaginary time G
                update_Gτ0!(Gτ0, γ, Gτ, j, ws)
                update_G0τ!(G0τ, γ, Gτ, j, ws)
                # update Gτ, standard
                update_G!(Gτ, γ, 1.0, j, ws)
                # update Grover inverse
                update_invGA!(replica, ρ)
            end
        end
        
        # compute G <- Bk⁻¹ * G * Bk to restore the ordering
        system.useFirstOrderTrotter || begin 
                                        wrap_G!(Gτ, Bk⁻¹, Bk, ws)
                                        wrap_G!(Gτ0, Bk⁻¹, Bk, ws)
                                        wrap_G!(G0τ, Bk⁻¹, Bk, ws)
                                    end

        @views σ = walker.auxfield[:, l]
        imagtime_propagator!(Bl[i], σ, system, tmpmat = ws.M)

        ### proceed to next time slice ###
        wrap_Gs!(Gτ, Gτ0, G0τ, Bl[i], ws)
    end

    #@views copyto!(cluster.B[cidx], prod(Bl[k:-1:1]))
    @views prod_cluster!(cluster.B[cidx], Bl[k:-1:1], ws.M)

    return nothing
end

function sweep!(
    system::Hubbard, qmc::QMC, 
    replica::Replica{W, ComplexF64, Float64},
    walker::W, ridx::Int;
    isJumpReplica::Bool=true
) where {W<:GCWalker}
    """
        sweep!(system, qmc, replica, walker, ridx)

        Sweep a replica (two walkers) over the imaginary time from 0 to β (ridx=1) or from β to 2β (ridx=2)
    """
    ### set alias ###
    K = qmc.K
    Aidx = replica.Aidx
    ws = walker.ws
    Bc = walker.Bc.B
    logdetGA, sgnlogdetGA = replica.logdetGA, replica.sgnlogdetGA
    # temporal factorizations
    tmpL = walker.FC.B
    Bτ = walker.Fτ[1]
    tmpM = walker.F
    # imaginary-time-displaced Green's
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]

    ridx == 1 ? (G₀ = replica.G₀1; G₀′ = replica.G₀2) : 
                (G₀ = replica.G₀2; G₀′ = replica.G₀1)

    @inbounds for cidx in 1 : K
        # then update a cluster of fields
        update_cluster!(walker, replica, system, qmc, cidx, ridx)

        # multiply the updated slice to the right factorization
        lmul!(Bc[cidx], Bτ, ws)

        # recompute the Grover inverse
        mul!(tmpM[1], tmpL[cidx], Bτ, ws)
        inv_IpA!(G₀, tmpM[1], ws)
        ridx == 1 ? begin
                logdetGA[], sgnlogdetGA[] =  @views inv_Grover!(replica.GA⁻¹, G₀[Aidx, Aidx], G₀′[Aidx, Aidx], replica.ws)
            end :
            begin
                logdetGA[], sgnlogdetGA[] =  @views inv_Grover!(replica.GA⁻¹, G₀′[Aidx, Aidx], G₀[Aidx, Aidx], replica.ws)
            end
        
        # G needs to be periodically recomputed from scratch
        mul!(tmpM[1], Bτ, tmpL[cidx], ws)
        update!(walker, identicalSpin=true)

        # compute imaginary-time-displaced Green's
        inv_invUpV!(Gτ0, Bτ, tmpL[cidx], ws)
        inv_invUpV!(G0τ, tmpL[cidx], Bτ, ws)
        @. G0τ *=-1
    end

    # At the end of the simulation, recompute all partial factorizations
    build_propagator!(walker.FC.B ,walker.Bc, walker.ws, isSymmetric=true)

    # save Bτ
    copyto!(walker.F[1], Bτ)
    # then reset Bτ to unit matrix
    ldr!(Bτ, I)

    isJumpReplica && begin
       # switch the matrix I - 2*GA to the next replica
        Im2GA = replica.Im2GA
        for i in CartesianIndices(Im2GA)
            @inbounds Im2GA[i] = -2 * G₀[i]
        end
        Im2GA[diagind(Im2GA)] .+= 1 
    end

    return nothing
end
