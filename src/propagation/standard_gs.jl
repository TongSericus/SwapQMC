"""
    Projective Monte Carlo Propagation in the regular Z space
"""

###################################################
##### Symmetric Sweep for Charge HS Transform #####
###################################################
function local_update!_symmetric(
    σ::AbstractArray{Int}, j::Int, l::Int, dir::Int, walker::HubbardWalker;
    useHeatbath::Bool = true, saveRatio::Bool = true
)
    
    α = walker.α
    G = walker.G[1]
    ws = walker.ws

    σj = flip_HSField(σ[j])
    # compute ratios of determinants through G
    r, d = compute_Metropolis_ratio(G, α[1, σj], j)
    saveRatio && push!(walker.tmp_r, r)
    # accept ratio
    u = useHeatbath ? real(r) / (1 + real(r)) : real(r)

    if rand() < u
        # accept the move, update the field and the Green's function
        walker.auxfield[j, l] *= -1
        update_G!(G, α[1, σj], d, j, ws, direction=dir)
    end
end

function update_cluster!_symmetric(
    walker::HubbardWalker,
    system::Hubbard, qmc::QMC, cidx::Int;
    direction::Int = 1
)
    k = qmc.K_interval[cidx]

    Bk = system.Bk
    Bk⁻¹ = system.Bk⁻¹

    G = walker.G[1]
    ws = walker.ws
    Bl = walker.Bl.B
    Bc = walker.Bc[1]

    # propagate from τ to τ+k
    direction == 1 && begin
        for i in 1 : k
            l = (cidx - 1) * qmc.stab_interval + i
            @views σ = walker.auxfield[:, l]

            # compute G <- Bk * G * Bk⁻¹ to enable fast update
            system.useFirstOrderTrotter || wrap_G!(G, Bk, Bk⁻¹, ws)

            # local updates
            for j in 1 : system.V
                local_update!_symmetric(σ, j, l, 1, walker, 
                                        saveRatio=qmc.saveRatio, 
                                        useHeatbath=qmc.useHeatbath
                                    )
            end
        
            # compute G <- Bk⁻¹ * G * Bk to restore the ordering
            system.useFirstOrderTrotter || wrap_G!(G, Bk⁻¹, Bk, ws)

            @views σ = walker.auxfield[:, l]
            imagtime_propagator!(Bl[i], σ, system, tmpmat = ws.M)

            # rank-1 update of the Green's function
            wrap_G!(G, Bl[i], ws, direction=1)
        end
        
        @views prod_cluster!(Bc, Bl[k:-1:1], ws.M)

        return nothing
    end

    # propagate from τ+k to τ
    for i in k:-1:1
        
        l = (cidx - 1) * qmc.stab_interval + i
        @views σ = walker.auxfield[:, l]

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || wrap_G!(G, Bk⁻¹, Bk, ws)

        # local updates
        for j in 1 : system.V
            local_update!_symmetric(σ, j, l, 2, walker, 
                                        saveRatio=qmc.saveRatio, 
                                        useHeatbath=qmc.useHeatbath
                                    )
        end
    
        # compute G <- Bk⁻¹ * G * Bk to restore the ordering
        system.useFirstOrderTrotter || wrap_G!(G, Bk, Bk⁻¹, ws)

        @views σ = walker.auxfield[:, l]
        imagtime_propagator!(Bl[i], σ, system, tmpmat = ws.M)

        # rank-1 update of the Green's function
        wrap_G!(G, Bl[i], ws, direction=2)
    end

    @views prod_cluster!(Bc, Bl[k:-1:1], ws.M)

    return nothing
end

function sweep!_symmetric(
    system::Hubbard, qmc::QMC, 
    walker::HubbardWalker, slice::UnitRange{Int}
)
    direction = slice.start < slice.stop ? 1 : 2

    K = qmc.K

    ws = walker.ws
    Bc = walker.Bc[1]
    Fl = walker.Fl[1]
    Fr = walker.Fr[1]
    Fcl = walker.Fcl.B
    Fcr = walker.Fcr.B

    # propagate from θ to 2θ or from 0 to θ
    direction == 1 && begin 
        for (i, cidx) in enumerate(slice)
            update_cluster!_symmetric(walker, system, qmc, cidx, direction=1)

            # multiply the updated slice to the right factorization on the left
            lmul!(Bc, Fr, ws)
            # save the multiplied results to the partial factorizations
            copyto!(Fcr[end-i+1-K], Fr)
            # move the left factorization to the next cluster
            copyto!(Fl, Fcl[i+1])

            # G needs to be periodically recomputed from scratch
            compute_G!(walker, 1, Bl=Fcl[end-i+1-K])
        end
        copyto!(walker.G[2], walker.G[1])

        return nothing
    end

    # propagate from 2θ to θ or from θ to 0
    for cidx in slice
        update_cluster!_symmetric(walker, system, qmc, cidx, direction=2)

        # multiply the updated slice to the left factorization on the right
        rmul!(Fl, Bc, ws)
        # save the multiplied results to the partial factorizations
        copyto!(Fcl[end-i+1-K], Fl)

        # G needs to be periodically recomputed from scratch
        compute_G!(walker, 1, Br=Fcr[end-i+1-K])
    end

    copyto!(walker.G[2], walker.G[1])

    return nothing
end
