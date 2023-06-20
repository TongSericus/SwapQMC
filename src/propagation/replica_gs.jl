"""
    Replica Monte Carlo sweep in the Z_{A, 2} space, ground state
"""

###################################################
##### Symmetric Sweep for Charge HS Transform #####
###################################################
function local_update!_symmetric(
    σ::AbstractArray{Int}, j::Int, l::Int, ridx::Int, dir::Int, 
    walker::HubbardWalker, replica::Replica;
    useHeatbath::Bool = true, saveRatio::Bool = true
)
    α = walker.α
    Gτ = walker.G[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    ws = walker.ws

    σj = flip_HSField(σ[j])
    # compute ratios of determinants through G
    r, γ, ρ = compute_Metropolis_ratio(system, replica, walker, α[1, σj], j, ridx)
    saveRatio && push!(walker.tmp_r, r)
    # accept ratio
    u = useHeatbath ? real(r) / (1 + real(r)) : real(r)
    
    if rand() < u
        # accept the move, update the field and the Green's function
        walker.auxfield[j, l] *= -1
        
        ### rank-1 updates ###
        # update imaginary time G
        update_Gτ0!(Gτ0, γ, Gτ, j, ws)
        update_G0τ!(G0τ, γ, Gτ, j, ws)
        # update Gτ, standard
        update_G!(Gτ, γ, 1.0, j, ws, direction=dir)
        # update Grover inverse
        update_invGA!(replica, ρ)
    end
end

function update_cluster!_symmetric(
    walker::HubbardWalker, replica::Replica,
    system::Hubbard, qmc::QMC, cidx::Int, ridx::Int
)
    k = qmc.K_interval[cidx]

    Bk = system.Bk
    Bk⁻¹ = system.Bk⁻¹

    Gτ = walker.G[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    ws = walker.ws
    Bl = walker.Bl.B
    Bc = walker.Bc[1]

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
               local_update!_symmetric(σ, j, l, ridx, 1, walker, replica,
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
            wrap_Gs!(Gτ, Gτ0, G0τ, Bl[i], ws)
        end

        @views prod_cluster!(Bc, Bl[k:-1:1], ws.M)

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
           local_update!_symmetric(σ, j, l, ridx, 2, walker, replica,
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
        wrap_Gs!(Gτ, Gτ0, G0τ, Bl[i], ws)
    end

    @views prod_cluster!(Bc, Bl[k:-1:1], ws.M)

    return nothing
end