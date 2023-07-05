"""
    Measure local observables right after the space and time lattice positions of its components are updated

    See PhysRevE.106.025318 for more details
"""

function sweep!(
    system::Hubbard, qmc::QMC, walker::HubbardWalker, 
    sampler::EtgSampler;
    loop_number::Int = 1
)
    Θ = div(qmc.K,2)

    if system.useChargeHST || qmc.forceSymmetry
        for i in 1 : loop_number
            sweep!_symmetric(system, qmc, walker, collect(Θ+1:2Θ))
            sweep!_symmetric(system, qmc, walker, collect(2Θ:-1:Θ+1))
            sweep!_symmetric(system, qmc, walker, sampler, collect(Θ:-1:1))
            sweep!_symmetric(system, qmc, walker, collect(1:Θ))
        end

        return nothing
    end
end

###################################################
##### Symmetric Sweep with Local Measurements #####
###################################################
function update_cluster!_symmetric(
    walker::HubbardWalker, 
    #replica::Replica, sampler::EtgSampler,
    sampler::EtgSampler,
    system::Hubbard, qmc::QMC, cidx::Int;
    direction::Int = 1
)
    k = qmc.K_interval[cidx]

    direction == 1 ? (
        # propagate from τ to τ+k
        Bk = system.Bk;
        Bk⁻¹ = system.Bk⁻¹;
        slice = collect(1:k)
    ) : 
    (
        # propagate from τ+k to τ
        Bk = system.Bk⁻¹;
        Bk⁻¹ = system.Bk;
        slice = collect(k:-1:1)
    )

    G = walker.G[1]
    ws = walker.ws
    Bl = walker.Bl.B
    Bc = walker.Bc.B[1]

    for i in slice
        l = (cidx - 1) * qmc.stab_interval + i
        @views σ = walker.auxfield[:, l]

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || wrap_G!(G, Bk, Bk⁻¹, ws)

        # local updates
        for j in 1 : system.V
            local_update!_symmetric(σ, j, l, walker,
                                    direction=direction,
                                    forceSymmetry=qmc.forceSymmetry,
                                    saveRatio=qmc.saveRatio, 
                                    useHeatbath=qmc.useHeatbath
                                )
            # make local measurements
            if l == sampler.mp_t && j == sampler.mp_x
                sampler.m_counter[] += 1
                if sampler.m_counter[] == qmc.measure_interval
                    wrap_G!(G, Bk⁻¹, Bk, ws)
                    measure_Pn!(sampler, walker)
                    wrap_G!(G, Bk, Bk⁻¹, ws)
                end
            end
        end
        
        # compute G <- Bk⁻¹ * G * Bk to restore the ordering
        system.useFirstOrderTrotter || wrap_G!(G, Bk⁻¹, Bk, ws)

        @views σ = walker.auxfield[:, l]
        imagtime_propagator!(Bl[i], σ, system, tmpmat = ws.M)

        # rank-1 update of the Green's function
        wrap_G!(G, Bl[i], ws, direction=direction)
    end
        
    @views prod_cluster!(Bc, Bl[k:-1:1], ws.M)

    return nothing
end

function sweep!_symmetric(
    system::Hubbard, qmc::QMC, 
    walker::HubbardWalker, 
    #replica::Replica, sampler::EtgSampler,
    sampler::EtgSampler,
    slice::Vector{Int}
)
    direction = slice[1] < slice[end] ? 1 : 2

    Θ = div(qmc.K,2)

    ws = walker.ws
    Bc = walker.Bc.B[1]
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
        qmc.forceSymmetry && conj!(walker.G[2])

        return nothing
    end

    # propagate from 2θ to θ or from θ to 0
    for (i, cidx) in zip(Iterators.reverse(eachindex(slice)), slice)
        #update_cluster!_symmetric(walker, replica, sampler, system, qmc, cidx, direction=2)
        update_cluster!_symmetric(walker, sampler, system, qmc, cidx, direction=2)

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
    qmc.forceSymmetry && conj!(walker.G[2])

    return nothing
end
