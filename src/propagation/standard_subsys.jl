"""
    Projective Monte Carlo Propagation in the regular Z space
"""

function sweep!(
    system::Hubbard, qmc::QMC, walker::HubbardSubsysWalker; 
    loop_number::Int=1
)
    Θ = div(qmc.K,2)

    if system.useChargeHST
        for i in 1 : loop_number
            sweep!_symmetric(system, qmc, walker, collect(Θ+1:2Θ))
            sweep!_symmetric(system, qmc, walker, collect(Θ:-1:1))
        end

        return nothing
    end
end

function sweep!(
    system::Hubbard, qmc::QMC, walker::HubbardSubsysWalker, sampler::EtgSampler; 
    loop_number::Int=1
)
    Θ = div(qmc.K,2)

    if system.useChargeHST
        for i in 1 : loop_number
            sweep!_symmetric(system, qmc, walker, sampler, collect(Θ+1:2Θ))
            sweep!_symmetric(system, qmc, walker, sampler, collect(Θ:-1:1))
        end

        return nothing
    end
end

###################################################
##### Symmetric Sweep for Charge HS Transform #####
###################################################
function local_update!_symmetric(
    σ::AbstractArray{Int}, j::Int, l::Int, 
    system::Hubbard, walker::HubbardSubsysWalker;
    direction::Int = 1,
    useHeatbath::Bool = true, saveRatio::Bool = true
)
    
    α = walker.α
    G = walker.G[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    ws = walker.ws

    σj = flip_HSField(σ[j])
    # compute ratios of determinants through G
    r, γ, ρ = compute_Metropolis_ratio(
                system, walker, α[1, σj], j, 
                direction=direction
            )
    saveRatio && push!(walker.tmp_r, r)
    # accept ratio
    u = useHeatbath ? real(r) / (1 + real(r)) : real(r)

    if rand() < u
        # accept the move, update the field and the Green's function
        walker.auxfield[j, l] *= -1
        ### rank-1 updates ###
        # update imaginary time G
        update_Gτ0!(Gτ0, γ, G, j, ws, direction=direction)
        update_G0τ!(G0τ, γ, G, j, ws, direction=direction)
        # update G, standard
        update_G!(G, γ, 1.0, j, ws, direction=direction)
        # update (I - GA)⁻¹      
        update_invImGA!(walker, ρ)
    end
end

function update_cluster!_symmetric(
    walker::HubbardSubsysWalker,
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

    # all Green's functions
    G = walker.G[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    gτ0 = walker.gτ0
    g0τ = walker.g0τ

    ws = walker.ws
    Bl = walker.Bl.B
    Bc = walker.Bc[1]

    for i in slice

        l = (cidx - 1) * qmc.stab_interval + i
        @views σ = walker.auxfield[:, l]

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || begin 
                                        wrap_G!(G, Bk, Bk⁻¹, ws)
                                        wrap_G!(Gτ0, Bk, Bk⁻¹, ws)
                                        wrap_G!(G0τ, Bk, Bk⁻¹, ws)
                                    end

        # local updates
        for j in 1 : system.V
            local_update!_symmetric(σ, j, l, system, walker,
                                    direction=direction, 
                                    saveRatio=qmc.saveRatio, 
                                    useHeatbath=qmc.useHeatbath
                                )
        end
    
        # compute G <- Bk⁻¹ * G * Bk to restore the ordering
        system.useFirstOrderTrotter || begin 
                                        wrap_G!(G, Bk⁻¹, Bk, ws)
                                        wrap_G!(Gτ0, Bk⁻¹, Bk, ws)
                                        wrap_G!(G0τ, Bk⁻¹, Bk, ws)
                                    end

        @views σ = walker.auxfield[:, l]
        imagtime_propagator!(Bl[i], σ, system, tmpmat = ws.M)

        # rank-1 update of the Green's function
        wrap_Gs!(G, Gτ0, G0τ, Bl[i], ws, direction=direction)
    end

    @views prod_cluster!(Bc, Bl[k:-1:1], ws.M)
    proceed_gτ0!(gτ0[cidx], Bc, G, ws, direction=direction)
    proceed_g0τ!(g0τ[cidx], Bc, G, ws, direction=direction)

    return nothing
end

function sweep!_symmetric(
    system::Hubbard, qmc::QMC, 
    walker::HubbardSubsysWalker, slice::Vector{Int}
)
    direction = slice[1] <= slice[end] ? 1 : 2

    ### set alias ###
    Θ = div(qmc.K,2)
    ws = walker.ws
    Bc = walker.Bc[1]
    φ₀ = walker.φ₀[1]
    φ₀T = walker.φ₀T[1]
    # temporal factorizations
    Fτt, FτT, _ = walker.Fτ
    Fl = walker.Fl[1]
    Fr = walker.Fr[1]
    Fcl = walker.Fcl.B
    Fcr = walker.Fcr.B

    # All Green's
    G = walker.G[1]
    G₀ = walker.G₀[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    gτ0 = walker.gτ0
    g0τ = walker.g0τ

    # subsystem-related quantities
    Aidx = walker.Aidx
    ImGA⁻¹ = walker.ImGA⁻¹[1]

    # propagate from θ to 2θ
    direction == 1 && begin
        for (i, cidx) in enumerate(slice)
            update_cluster!_symmetric(walker, system, qmc, cidx, direction=1)

            # multiply the updated slice to the right factorization on the left
            lmul!(Bc, Fτt, ws)

            # G needs to be periodically recomputed from scratch
            mul!(FτT, Fτt, Fr, ws)
            compute_G!(walker, 1, Bl=Fcl[i], Br=FτT)

            # recompute imaginary-time-displaced Green's
            @views prod_cluster!(Gτ0, gτ0[cidx:-1:Θ+1], ws.M)
            @views prod_cluster!(G0τ, g0τ[Θ+1:cidx], ws.M)
            (cidx - Θ - 1) % 2 == 0 || @. G0τ *= -1

            # recompute G₀
            mul!(FτT, Fcl[i], Fτt, ws)
            compute_G!(G₀, φ₀, φ₀T, walker.Ul, walker.Ur, FτT, Fr)

            # recompute (I - GA)⁻¹
            @views compute_invImGA!(ImGA⁻¹, G₀[Aidx, Aidx], walker.wsA)

            cidx == 2Θ ? (
                    copyto!(Fl, FτT); copyto!(G, G₀); 
                    copyto!(Gτ0, G);
                    copyto!(G0τ, G);
                    G0τ[diagind(G0τ)] .-= 1
                ) : 
                    mul!(Fcl[i], Fcl[i+1], Bc, ws)
        end

        # reset temporal factorizations
        ldr!(Fτt, I)
        ldr!(FτT, I)

        # copy green's function to the spin-down sector
        copyto!(walker.G[2], walker.G[1])

        return nothing
    end

    # propagate from θ to 0
    for (i, cidx) in zip(Iterators.reverse(eachindex(slice)), slice)
        update_cluster!_symmetric(walker, system, qmc, cidx, direction=2)

        # multiply the updated slice to the left factorization on the right
        rmul!(Fτt, Bc, ws)

        # G needs to be periodically recomputed from scratch
        mul!(FτT, Fl, Fτt, ws)
        compute_G!(walker, 1, Bl=FτT, Br=Fcr[i])
        
        # recompute imaginary-time-displaced Green's
        @views prod_cluster!(Gτ0, gτ0[Θ:-1:cidx], ws.M)
        @views prod_cluster!(G0τ, g0τ[cidx:Θ], ws.M)
        (Θ - cidx) % 2 == 0 || @. G0τ *= -1

        # recompute G₀
        mul!(FτT, Fτt, Fcr[i], ws)
        compute_G!(G₀, φ₀, φ₀T, walker.Ul, walker.Ur, Fl, FτT)

        # recompute (I - GA)⁻¹
        @views compute_invImGA!(ImGA⁻¹, G₀[Aidx, Aidx], walker.wsA)

        cidx == 1 ? (
                copyto!(Fr, FτT);
                copyto!(G, G₀);
                copyto!(Gτ0, G);
                copyto!(G0τ, G);
                G0τ[diagind(G0τ)] .-= 1
            ) : 
                mul!(Fcr[i], Bc, Fcr[i-1], ws)
    end

    # reset temporal factorizations
    ldr!(Fτt, I)
    ldr!(FτT, I)

    # copy green's function to the spin-down sector
    copyto!(walker.G[2], walker.G[1])

    return nothing
end

###################################################
##### Symmetric Sweep with Local Measurements #####
###################################################
function update_cluster!_symmetric(
    walker::HubbardSubsysWalker, sampler::EtgSampler,
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

    # all Green's functions
    G = walker.G[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    gτ0 = walker.gτ0
    g0τ = walker.g0τ

    ws = walker.ws
    Bl = walker.Bl.B
    Bc = walker.Bc[1]

    for i in slice

        l = (cidx - 1) * qmc.stab_interval + i
        @views σ = walker.auxfield[:, l]

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || begin 
                                        wrap_G!(G, Bk, Bk⁻¹, ws)
                                        wrap_G!(Gτ0, Bk, Bk⁻¹, ws)
                                        wrap_G!(G0τ, Bk, Bk⁻¹, ws)
                                    end

        # local updates
        for j in 1 : system.V
            local_update!_symmetric(σ, j, l, system, walker,
                                    direction=direction, 
                                    saveRatio=qmc.saveRatio, 
                                    useHeatbath=qmc.useHeatbath
                                )
            # make local measurements
            if l == sampler.mp_t && j == sampler.mp_x
                sampler.m_counter[] += 1
                if sampler.m_counter[] == qmc.measure_interval
                    wrap_G!(G, Bk⁻¹, Bk, ws)
                    measure!(sampler, walker)
                    wrap_G!(G, Bk, Bk⁻¹, ws)
                end
            end
        end
    
        # compute G <- Bk⁻¹ * G * Bk to restore the ordering
        system.useFirstOrderTrotter || begin 
                                        wrap_G!(G, Bk⁻¹, Bk, ws)
                                        wrap_G!(Gτ0, Bk⁻¹, Bk, ws)
                                        wrap_G!(G0τ, Bk⁻¹, Bk, ws)
                                    end

        @views σ = walker.auxfield[:, l]
        imagtime_propagator!(Bl[i], σ, system, tmpmat = ws.M)

        # rank-1 update of the Green's function
        wrap_Gs!(G, Gτ0, G0τ, Bl[i], ws, direction=direction)
    end

    @views prod_cluster!(Bc, Bl[k:-1:1], ws.M)
    proceed_gτ0!(gτ0[cidx], Bc, G, ws, direction=direction)
    proceed_g0τ!(g0τ[cidx], Bc, G, ws, direction=direction)

    return nothing
end

function sweep!_symmetric(
    system::Hubbard, qmc::QMC, 
    walker::HubbardSubsysWalker, sampler::EtgSampler,
    slice::Vector{Int}
)
    direction = slice[1] <= slice[end] ? 1 : 2

    ### set alias ###
    Θ = div(qmc.K,2)
    ws = walker.ws
    Bc = walker.Bc[1]
    φ₀ = walker.φ₀[1]
    φ₀T = walker.φ₀T[1]
    # temporal factorizations
    Fτt, FτT, _ = walker.Fτ
    Fl = walker.Fl[1]
    Fr = walker.Fr[1]
    Fcl = walker.Fcl.B
    Fcr = walker.Fcr.B

    # All Green's
    G = walker.G[1]
    G₀ = walker.G₀[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    gτ0 = walker.gτ0
    g0τ = walker.g0τ

    # subsystem-related quantities
    Aidx = walker.Aidx
    ImGA⁻¹ = walker.ImGA⁻¹[1]

    # propagate from θ to 2θ
    direction == 1 && begin
        for (i, cidx) in enumerate(slice)
            update_cluster!_symmetric(walker, sampler, system, qmc, cidx, direction=1)

            # multiply the updated slice to the right factorization on the left
            lmul!(Bc, Fτt, ws)

            # G needs to be periodically recomputed from scratch
            mul!(FτT, Fτt, Fr, ws)
            compute_G!(walker, 1, Bl=Fcl[i], Br=FτT)

            # recompute imaginary-time-displaced Green's
            @views prod_cluster!(Gτ0, gτ0[cidx:-1:Θ+1], ws.M)
            @views prod_cluster!(G0τ, g0τ[Θ+1:cidx], ws.M)
            (cidx - Θ - 1) % 2 == 0 || @. G0τ *= -1

            # recompute G₀
            mul!(FτT, Fcl[i], Fτt, ws)
            compute_G!(G₀, φ₀, φ₀T, walker.Ul, walker.Ur, FτT, Fr)

            # recompute (I - GA)⁻¹
            @views compute_invImGA!(ImGA⁻¹, G₀[Aidx, Aidx], walker.wsA)

            cidx == 2Θ ? (
                    copyto!(Fl, FτT); copyto!(G, G₀); 
                    copyto!(Gτ0, G);
                    copyto!(G0τ, G);
                    G0τ[diagind(G0τ)] .-= 1
                ) : 
                    mul!(Fcl[i], Fcl[i+1], Bc, ws)
        end

        # reset temporal factorizations
        ldr!(Fτt, I)
        ldr!(FτT, I)

        # copy green's function to the spin-down sector
        copyto!(walker.G[2], walker.G[1])

        return nothing
    end

    # propagate from θ to 0
    for (i, cidx) in zip(Iterators.reverse(eachindex(slice)), slice)
        update_cluster!_symmetric(walker, sampler, system, qmc, cidx, direction=2)

        # multiply the updated slice to the left factorization on the right
        rmul!(Fτt, Bc, ws)

        # G needs to be periodically recomputed from scratch
        mul!(FτT, Fl, Fτt, ws)
        compute_G!(walker, 1, Bl=FτT, Br=Fcr[i])
        
        # recompute imaginary-time-displaced Green's
        @views prod_cluster!(Gτ0, gτ0[Θ:-1:cidx], ws.M)
        @views prod_cluster!(G0τ, g0τ[cidx:Θ], ws.M)
        (Θ - cidx) % 2 == 0 || @. G0τ *= -1

        # recompute G₀
        mul!(FτT, Fτt, Fcr[i], ws)
        compute_G!(G₀, φ₀, φ₀T, walker.Ul, walker.Ur, Fl, FτT)

        # recompute (I - GA)⁻¹
        @views compute_invImGA!(ImGA⁻¹, G₀[Aidx, Aidx], walker.wsA)

        cidx == 1 ? (
                copyto!(Fr, FτT);
                copyto!(G, G₀);
                copyto!(Gτ0, G);
                copyto!(G0τ, G);
                G0τ[diagind(G0τ)] .-= 1
            ) : 
                mul!(Fcr[i], Bc, Fcr[i-1], ws)
    end

    # reset temporal factorizations
    ldr!(Fτt, I)
    ldr!(FτT, I)

    # copy green's function to the spin-down sector
    copyto!(walker.G[2], walker.G[1])

    return nothing
end