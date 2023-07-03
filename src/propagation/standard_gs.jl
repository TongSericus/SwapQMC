"""
    Projective Monte Carlo Propagation in the regular Z space
"""

function sweep!(system::Hubbard, qmc::QMC, walker::HubbardWalker; loop_number::Int = 1)
    Θ = div(qmc.K,2)

    if system.useChargeHST || qmc.forceSymmetry  # charge decomposition
        for i in 1 : loop_number
            sweep!_symmetric(system, qmc, walker, collect(Θ+1:2Θ))
            sweep!_symmetric(system, qmc, walker, collect(2Θ:-1:Θ+1))
            sweep!_symmetric(system, qmc, walker, collect(Θ:-1:1))
            sweep!_symmetric(system, qmc, walker, collect(1:Θ))
        end

        return nothing
    else                                        # spin decomposition
        for i in 1 : loop_number
            sweep!_asymmetric(system, qmc, walker, collect(Θ+1:2Θ))
            sweep!_asymmetric(system, qmc, walker, collect(2Θ:-1:Θ+1))
            sweep!_asymmetric(system, qmc, walker, collect(Θ:-1:1))
            sweep!_asymmetric(system, qmc, walker, collect(1:Θ))
        end

        return nothing
    end
end

###################################################
##### Symmetric Sweep for Charge HS Transform #####
###################################################
function local_update!_symmetric(
    σ::AbstractArray{Int}, j::Int, l::Int, walker::HubbardWalker;
    direction::Int = 1, forceSymmetry::Bool = false,
    useHeatbath::Bool = true, saveRatio::Bool = true
)
    
    α = walker.α
    G = walker.G[1]
    ws = walker.ws

    σj = flip_HSField(σ[j])
    # compute ratios of determinants through G
    r, d = compute_Metropolis_ratio(G, α[1, σj], j, forceSymmetry=forceSymmetry)
    saveRatio && push!(walker.tmp_r, r)
    # accept ratio
    u = useHeatbath ? real(r) / (1 + real(r)) : real(r)

    if rand() < u
        # accept the move, update the field and the Green's function
        walker.auxfield[j, l] *= -1
        update_G!(G, α[1, σj], d, j, ws, direction=direction)
    end
end

function update_cluster!_symmetric(
    walker::HubbardWalker,
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
    walker::HubbardWalker, slice::Vector{Int}
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
    qmc.forceSymmetry && conj!(walker.G[2])

    return nothing
end

##################################################
##### Asymmetric Sweep for Spin HS Transform #####
##################################################
function local_update!_asymmetric(
    σ::AbstractArray{Int}, j::Int, l::Int, walker::HubbardWalker;
    direction::Int = 1,
    useHeatbath::Bool = true, saveRatio::Bool = true
)
    
    α = walker.α
    G = walker.G
    ws = walker.ws

    σj = flip_HSField(σ[j])
    # compute ratios of determinants through G
    r, d_up, d_dn = compute_Metropolis_ratio(G, α, σj, j)
    saveRatio && push!(walker.tmp_r, r)
    # accept ratio
    u = useHeatbath ? real(r) / (1 + real(r)) : real(r)

    if rand() < u
        # accept the move, update the field and the Green's function
        walker.auxfield[j, l] *= -1
        update_G!(G[1], α[1, σj], d_up, j, ws, direction=direction)
        update_G!(G[2], α[2, σj], d_dn, j, ws, direction=direction)
    end
end

function update_cluster!_asymmetric(
    walker::HubbardWalker,
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

    G = walker.G
    ws = walker.ws
    Bl = walker.Bl.B
    Bc = @view walker.Bc.B[1:2]

    for i in slice
        l = (cidx - 1) * qmc.stab_interval + i
        @views σ = walker.auxfield[:, l]

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || (wrap_G!(G[1], Bk, Bk⁻¹, ws); wrap_G!(G[2], Bk, Bk⁻¹, ws))

        # local updates
        for j in 1 : system.V
            local_update!_asymmetric(σ, j, l, walker,
                                    direction=direction,
                                    saveRatio=qmc.saveRatio, 
                                    useHeatbath=qmc.useHeatbath
                                )
        end
        
        # compute G <- Bk⁻¹ * G * Bk to restore the ordering
        system.useFirstOrderTrotter || (wrap_G!(G[1], Bk⁻¹, Bk, ws); wrap_G!(G[2], Bk⁻¹, Bk, ws))

        @views σ = walker.auxfield[:, l]
        imagtime_propagator!(Bl[i], Bl[k+i], σ, system, tmpmat = ws.M)

        # rank-1 update of the Green's function
        wrap_G!(G[1], Bl[i], ws, direction=direction)
        wrap_G!(G[2], Bl[k+i], ws, direction=direction)
    end
        
    @views prod_cluster!(Bc[1], Bl[k:-1:1], ws.M)
    @views prod_cluster!(Bc[2], Bl[2k:-1:k+1], ws.M)

    return nothing
end

function sweep!_asymmetric(
    system::Hubbard, qmc::QMC, 
    walker::HubbardWalker, slice::Vector{Int}
)
    direction = slice[1] < slice[end] ? 1 : 2

    Θ = div(qmc.K,2)

    ws = walker.ws
    Bc = @view walker.Bc.B[1:2]
    Fτ1, Fτ2, Fo1, Fo2 = walker.Fτ
    Fl = walker.Fl
    Fr = walker.Fr
    Fcl = walker.Fcl.B
    Fcr = walker.Fcr.B

    # propagate from θ to 2θ or from 0 to θ
    direction == 1 && begin
        for (i, cidx) in enumerate(slice)
            update_cluster!_asymmetric(walker, system, qmc, cidx, direction=1)

            # multiply the updated slice to the right factorization on the left
            copyto!(Fτ1, Fr[1])
            lmul!(Bc[1], Fr[1], ws)
            copyto!(Fτ2, Fr[2])
            lmul!(Bc[2], Fr[2], ws)

            if cidx > Θ         # sweeping the left configurations
                # G needs to be periodically recomputed from scratch
                compute_G!(walker, 1, Bl=Fcl[i])
                compute_G!(walker, 2, Bl=Fcl[Θ+i])
                # save the multiplied results to the partial factorizations
                copyto!(Fcl[i], Fτ1)
                copyto!(Fcl[Θ+i], Fτ2)
            elseif cidx <= Θ    # sweeping the right configurations
                # G needs to be periodically recomputed from scratch
                compute_G!(walker, 1, Bl=Fcr[i])
                compute_G!(walker, 2, Bl=Fcr[Θ+i])
                # save the multiplied results to the partial factorizations
                copyto!(Fcr[i], Fτ1)
                copyto!(Fcr[Θ+i], Fτ2)
            end
        end

        # reset left factorization at t=2θ
        slice[end] == qmc.K ? 
                    (
                        ldr!(Fl[1], I);
                        ldr!(Fl[2], I)
                    ) : 
                    (
                        copyto!(Fl[1], Fo1); copyto!(Fo1, Fr[1]); 
                        copyto!(Fl[2], Fo2); copyto!(Fo2, Fr[2])
                    )

        return nothing
    end

    # propagate from 2θ to θ or from θ to 0
    for (i, cidx) in zip(Iterators.reverse(eachindex(slice)), slice)
        update_cluster!_asymmetric(walker, system, qmc, cidx, direction=2)

        # multiply the updated slice to the left factorization on the right
        copyto!(Fτ1, Fl[1])
        rmul!(Fl[1], Bc[1], ws)
        copyto!(Fτ2, Fl[2])
        rmul!(Fl[2], Bc[2], ws)

        if cidx > Θ         # sweeping the left configurations
            # G needs to be periodically recomputed from scratch
            compute_G!(walker, 1, Br=Fcl[i])
            compute_G!(walker, 2, Br=Fcl[Θ+i])
            # save the multiplied results to the partial factorizations
            copyto!(Fcl[i], Fτ1)
            copyto!(Fcl[Θ+i], Fτ2)
        elseif cidx <= Θ    # sweeping the right configurations
            # G needs to be periodically recomputed from scratch
            compute_G!(walker, 1, Br=Fcr[i])
            compute_G!(walker, 2, Br=Fcr[Θ+i])
            # save the multiplied results to the partial factorizations
            copyto!(Fcr[i], Fτ1)
            copyto!(Fcr[Θ+i], Fτ2)
        end
    end

    # reset right factorization at t=0
    slice[end] == 1 ? 
                (
                    ldr!(Fr[1], I);
                    ldr!(Fr[2], I)
                ) : 
                (
                    copyto!(Fr[1], Fo1); copyto!(Fo1, Fl[1]);
                    copyto!(Fr[2], Fo2); copyto!(Fo2, Fl[2])
                )

    return nothing
end
