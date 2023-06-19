"""
    Projective Monte Carlo Propagation in the regular Z space
"""

function sweep!(system::Hubbard, qmc::QMC, walker::HubbardWalker)
    Θ = div(qmc.K,2)

    if system.useChargeHST
        sweep!_symmetric(system, qmc, walker, collect(Θ+1:2Θ))
        sweep!_symmetric(system, qmc, walker, collect(2Θ:-1:Θ+1))
        sweep!_symmetric(system, qmc, walker, collect(Θ:-1:1))
        sweep!_symmetric(system, qmc, walker, collect(1:Θ))

        return nothing
    end
end

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
    walker::HubbardWalker, slice::Vector{Int}
)
    direction = slice[1] < slice[end] ? 1 : 2

    Θ = div(qmc.K,2)

    ws = walker.ws
    Bc = walker.Bc[1]
    Fτ, F0, _ = walker.Fτ
    Fl = walker.Fl[1]
    Fr = walker.Fr[1]
    Fcl = walker.Fcl.B
    Fcr = walker.Fcr.B

    # propagate from θ to 2θ or from 0 to θ
    direction == 1 && begin
        for (i, cidx) in enumerate(slice)
            update_cluster!_symmetric(walker, system, qmc, cidx, direction=1)

            # multiply the updated slice to the right factorization on the left
            copyto!(Fτ, Fr)
            lmul!(Bc, Fr, ws)

            if cidx > Θ         # sweeping the left configurations
                # G needs to be periodically recomputed from scratch
                compute_G!(walker, 1, Bl=Fcl[i])
                # save the multiplied results to the partial factorizations
                copyto!(Fcl[i], Fτ)
            elseif cidx <= Θ    # sweeping the right configurations
                # G needs to be periodically recomputed from scratch
                compute_G!(walker, 1, Bl=Fcr[i])
                # save the multiplied results to the partial factorizations
                copyto!(Fcr[i], Fτ)
            end
        end

        # reset left factorization at t=2θ
        slice[end] == qmc.K ? ldr!(Fl, I) : (copyto!(Fl, F0); copyto!(F0, Fr))
        # copy green's function to the spin-down sector
        copyto!(walker.G[2], walker.G[1])

        return nothing
    end

    # propagate from 2θ to θ or from θ to 0
    for (i, cidx) in zip(Iterators.reverse(eachindex(slice)), slice)
        update_cluster!_symmetric(walker, system, qmc, cidx, direction=2)

        # multiply the updated slice to the left factorization on the right
        copyto!(Fτ, Fl)
        rmul!(Fl, Bc, ws)

        if cidx > Θ         # sweeping the left configurations
            # G needs to be periodically recomputed from scratch
            compute_G!(walker, 1, Br=Fcl[i])
            # save the multiplied results to the partial factorizations
            copyto!(Fcl[i], Fτ)
        elseif cidx <= Θ    # sweeping the right configurations
            # G needs to be periodically recomputed from scratch
            compute_G!(walker, 1, Br=Fcr[i])
            # save the multiplied results to the partial factorizations
            copyto!(Fcr[i], Fτ)
        end
    end

    # reset right factorization at t=0
    slice[end] == 1 ? ldr!(Fr, I) : (copyto!(Fr, F0); copyto!(F0, Fl))
    # copy green's function to the spin-down sector
    copyto!(walker.G[2], walker.G[1])

    return nothing
end
