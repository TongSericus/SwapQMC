"""
    Replica Monte Carlo sweep in the Z_{A, 2} space, ground state
"""

###################################################
##### Symmetric Sweep for Charge HS Transform #####
###################################################
function local_update!_symmetric(
    σ::AbstractArray{Int}, j::Int, l::Int, ridx::Int, 
    system::Hubbard, walker::HubbardWalker, replica::Replica;
    direction::Int = 1, region::Char = 'L',
    useHeatbath::Bool = true, saveRatio::Bool = true
)
    α = walker.α
    Gτ = walker.G[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    ws = walker.ws

    σj = flip_HSField(σ[j])
    # compute ratios of determinants through G
    r, γ, ρ = compute_Metropolis_ratio(
                system, replica, walker, α[1, σj], j, ridx,
                direction=direction, region=region
            )
    saveRatio && push!(walker.tmp_r, r)
    # accept ratio
    u = useHeatbath ? real(r) / (1 + real(r)) : real(r)
    
    if rand() < u
        # accept the move, update the field and the Green's function
        walker.auxfield[j, l] *= -1
        
        ### rank-1 updates ###
        # update imaginary time G
        if (direction == 2 && region == 'L') || (direction == 1 && region == 'R')
            update_Gτ0!(G0τ, γ, Gτ, j, ws, direction=direction)
            update_G0τ!(Gτ0, γ, Gτ, j, ws, direction=direction)
        else
            update_Gτ0!(Gτ0, γ, Gτ, j, ws, direction=direction)
            update_G0τ!(G0τ, γ, Gτ, j, ws, direction=direction)
        end
        # update Gτ, standard
        update_G!(Gτ, γ, 1.0, j, ws, direction=direction)
        # update Grover inverse
        update_invGA!(replica, ρ)
    end
end

function update_cluster!_symmetric(
    walker::HubbardWalker, replica::Replica,
    system::Hubbard, qmc::QMC, cidx::Int, ridx::Int;
    direction::Int = 1
)
    k = qmc.K_interval[cidx]

    Bk = system.Bk
    Bk⁻¹ = system.Bk⁻¹

    # all Green's functions
    Gτ = walker.G[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    gτ0 = walker.gτ0
    g0τ = walker.g0τ

    ws = walker.ws
    Bl = walker.Bl.B
    Bc = walker.Bc[1]

    Θ = div(qmc.K,2)
    cidx > Θ ? (reg = 'L') : (reg = 'R')

    # propagate from τ to τ+k
    direction == 1 && begin
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
               local_update!_symmetric(σ, j, l, ridx, 
                                       system, walker, replica,
                                       direction=1, region=reg,
                                       saveRatio=qmc.saveRatio,
                                       useHeatbath=qmc.useHeatbath
                                    )
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
            wrap_Gs!(Gτ, Gτ0, G0τ, Bl[i], ws, direction=1, region=reg)
        end

        @views prod_cluster!(Bc, Bl[k:-1:1], ws.M)
        proceed_gτ0!(gτ0[cidx], Bc, Gτ, ws, direction=1)
        proceed_g0τ!(g0τ[cidx], Bc, Gτ, ws, direction=1)

        return nothing
    end

    # propagate from τ+k to τ
    for i in k:-1:1

        l = (cidx - 1) * qmc.stab_interval + i
        @views σ = walker.auxfield[:, l]

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || begin 
                                        wrap_G!(Gτ, Bk⁻¹, Bk, ws)
                                        wrap_G!(Gτ0, Bk⁻¹, Bk, ws)
                                        wrap_G!(G0τ, Bk⁻¹, Bk, ws)
                                    end

        for j in 1 : system.V
           local_update!_symmetric(σ, j, l, ridx, 
                                   system, walker, replica,
                                   direction=2, region=reg,
                                   saveRatio=qmc.saveRatio,
                                   useHeatbath=qmc.useHeatbath
                                )
        end
    
        # compute G <- Bk⁻¹ * G * Bk to restore the ordering
        system.useFirstOrderTrotter || begin 
                                        wrap_G!(Gτ, Bk, Bk⁻¹, ws)
                                        wrap_G!(Gτ0, Bk, Bk⁻¹, ws)
                                        wrap_G!(G0τ, Bk, Bk⁻¹, ws)
                                    end

        @views σ = walker.auxfield[:, l]
        imagtime_propagator!(Bl[i], σ, system, tmpmat = ws.M)

        ### proceed to next time slice ###
        wrap_Gs!(Gτ, Gτ0, G0τ, Bl[i], ws, direction=2, region=reg)
    end

    @views prod_cluster!(Bc, Bl[k:-1:1], ws.M)
    proceed_gτ0!(gτ0[cidx], Bc, Gτ, ws, direction=2)
    proceed_g0τ!(g0τ[cidx], Bc, Gτ, ws, direction=2)

    return nothing
end

function sweep!_symmetric(
    system::Hubbard, qmc::QMC, 
    replica::Replica, walker::HubbardWalker,
    ridx::Int, slice::Vector{Int};
    jumpReplica::Bool=true
)
    direction = slice[1] < slice[end] ? 1 : 2
    ### set alias ###
    Θ = div(qmc.K,2)
    Aidx = replica.Aidx
    ws = walker.ws
    Bc = walker.Bc[1]
    logdetGA, sgnlogdetGA = replica.logdetGA, replica.sgnlogdetGA
    # temporal factorizations
    Fτ, Fτt, FτT, F0 = walker.Fτ
    Fl = walker.Fl[1]
    Fr = walker.Fr[1]
    Fcl = walker.Fcl.B
    Fcr = walker.Fcr.B
    # imaginary-time-displaced Green's
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    gτ0 = walker.gτ0
    g0τ = walker.g0τ

    ridx == 1 ? (G₀ = replica.G₀1; G₀′ = replica.G₀2) : 
                (G₀ = replica.G₀2; G₀′ = replica.G₀1)
    
    cidx > Θ ? copyto!(F0, Fr) : copyto!(F0, Fl)

    # propagate from θ to 2θ or from 0 to θ
    direction == 1 && begin
        for (i, cidx) in enumerate(slice)
            update_cluster!_symmetric(walker, replica, system, qmc, cidx, ridx, direction=1)

            # multiply the updated slice to the right factorization on the left
            lmul!(Bc, Fr, ws)

            if cidx > Θ         # sweeping the left configurations
                # Gτ needs to be periodically recomputed from scratch
                compute_G!(walker, 1, Bl=Fcl[i])
                # save the multiplied results to the partial factorizations
                lmul!(Bc, Fτt, ws)
                mul!(FτT, Fcl[i], Fτt, ws)
                copyto!(Fcl[i], Fτ)
                copyto!(Fτ, Fτt)

                # recompute imaginary-time-displaced Green's
                @views prod_cluster!(Gτ0, gτ0[cidx:-1:Θ+1], ws.M)
                @views prod_cluster!(G0τ, g0τ[Θ+1:cidx], ws.M)

                # recompute G₀
                compute_G!(G₀, walker.φ₀, walker.φ₀T, walker.Ul, walker.Ur, FτT, F0)

                # recompute Grover inverse
                ridx == 1 ? begin
                        logdetGA[], sgnlogdetGA[] =  @views inv_Grover!(replica.GA⁻¹, G₀[Aidx, Aidx], G₀′[Aidx, Aidx], replica.ws)
                    end :
                    begin
                        logdetGA[], sgnlogdetGA[] =  @views inv_Grover!(replica.GA⁻¹, G₀′[Aidx, Aidx], G₀[Aidx, Aidx], replica.ws)
                    end
            elseif cidx <= Θ    # sweeping the right configurations
                # G needs to be periodically recomputed from scratch
                mul!(FτT, Fl, Fcr[i])
                compute_G!(walker, 1, Bl=FτT)
                # save the multiplied results to the partial factorizations
                copyto!(Fcr[i], Fτ)
                # save the multiplied results to the partial factorizations
                lmul!(Bc, Fτt, ws)
                mul!(FτT, Fcr[i], Fτt)
                copyto!(Fcr[i], Fτ)
                copyto!(Fτ, Fτt)

                # recompute imaginary-time-displaced Green's
                @views prod_cluster!(Gτ0, gτ0[Θ:-1:cidx], ws.M)
                @views prod_cluster!(G0τ, g0τ[cidx:Θ], ws.M)

                # recompute G₀
                compute_G!(G₀, walker.φ₀, walker.φ₀T, walker.Ul, walker.Ur, F0, FτT)

                # recompute Grover inverse
                ridx == 1 ? begin
                        logdetGA[], sgnlogdetGA[] =  @views inv_Grover!(replica.GA⁻¹, G₀[Aidx, Aidx], G₀′[Aidx, Aidx], replica.ws)
                    end :
                    begin
                        logdetGA[], sgnlogdetGA[] =  @views inv_Grover!(replica.GA⁻¹, G₀′[Aidx, Aidx], G₀[Aidx, Aidx], replica.ws)
                    end
                end
        end

        # reset temporal factorizations
        ldr!(Fτ, I)
        ldr!(Fτt, I)
        # reset left factorization at t=2θ
        slice[end] == qmc.K ? ldr!(Fl, I) : (copyto!(Fl, F0); copyto!(F0, Fr))
        # copy green's function to the spin-down sector
        copyto!(walker.G[2], walker.G[1])
    end

    # propagate from 2θ to θ or from θ to 0
    for (i, cidx) in zip(Iterators.reverse(eachindex(slice)), slice)
        update_cluster!_symmetric(walker, replica, system, qmc, cidx, ridx, direction=2)

        # multiply the updated slice to the left factorization on the right
        rmul!(Fl, Bc, ws)

        if cidx > Θ         # sweeping the left configurations
            # G needs to be periodically recomputed from scratch
            mul!(FτT, Fcl[i], Fr)
            compute_G!(walker, 1, Br=FτT)
            # save the multiplied results to the partial factorizations
            rmul!(Fτt, Bc, ws)
            mul!(FτT, Fcl[i], Fτt)
            copyto!(Fcl[i], Fτ)
            copyto!(Fτ, Fτt)

            # recompute imaginary-time-displaced Green's
            @views prod_cluster!(Gτ0, gτ0[cidx+1:-1:Θ+1], ws.M)
            @views prod_cluster!(G0τ, g0τ[cidx+1:-1:Θ+1], ws.M)

            # recompute G₀
            compute_G!(G₀, walker.φ₀, walker.φ₀T, walker.Ul, walker.Ur, FτT, F0)

            # recompute Grover inverse
            ridx == 1 ? begin
                    logdetGA[], sgnlogdetGA[] =  @views inv_Grover!(replica.GA⁻¹, G₀[Aidx, Aidx], G₀′[Aidx, Aidx], replica.ws)
                end :
                begin
                    logdetGA[], sgnlogdetGA[] =  @views inv_Grover!(replica.GA⁻¹, G₀′[Aidx, Aidx], G₀[Aidx, Aidx], replica.ws)
                end
        elseif cidx <= Θ    # sweeping the right configurations
            # G needs to be periodically recomputed from scratch
            compute_G!(walker, 1, Br=Fcr[i])
            # save the multiplied results to the partial factorizations
            rmul!(Fτt, Bc, ws)
            mul!(FτT, Fτt, Fcr[i], ws)
            copyto!(Fcr[i], Fτ)
            copyto!(Fτ, Fτt)

            # recompute imaginary-time-displaced Green's
            @views prod_cluster!(Gτ0, gτ0[cidx+1:Θ+1], ws.M)
            @views prod_cluster!(G0τ, g0τ[cidx+1:Θ+1], ws.M)

            # recompute G₀
            compute_G!(G₀, walker.φ₀, walker.φ₀T, walker.Ul, walker.Ur, F0, FτT)

            # recompute Grover inverse
            ridx == 1 ? begin
                    logdetGA[], sgnlogdetGA[] =  @views inv_Grover!(replica.GA⁻¹, G₀[Aidx, Aidx], G₀′[Aidx, Aidx], replica.ws)
                end :
                begin
                    logdetGA[], sgnlogdetGA[] =  @views inv_Grover!(replica.GA⁻¹, G₀′[Aidx, Aidx], G₀[Aidx, Aidx], replica.ws)
                end
        end
    end
    
    jumpReplica && jump_replica!(replica.Im2GA, G₀, Aidx)

    return nothing
end

function jump_replica!(Im2GA::AbstractMatrix, G₀::AbstractMatrix, Aidx)
    @views G′ = G₀[Aidx, Aidx]
    for i in CartesianIndices(Im2GA)
        @inbounds Im2GA[i] = -2 * G′[i]
    end
    Im2GA[diagind(Im2GA)] .+= 1 

    return Im2GA
end
