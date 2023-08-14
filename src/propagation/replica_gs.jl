"""
    Replica Monte Carlo sweep in the Z_{A, 2} space, ground state
"""

"""
    sweep!(system, qmc, replica)

    Sweep a replica (two copies of walker) through the imaginary time, with the workflow
    1) update walker1 from θ to 2θ
    1) update walker2 from θ to 2θ
    1) update walker1 from θ to 0
    1) update walker2 from θ to 0
"""
function sweep!(system::Hubbard, qmc::QMC, replica::Replica)
    Θ = div(qmc.K,2)

    walker1 = replica.walker1
    walker2 = replica.walker2

    if system.useChargeHST
        
        sweep!_symmetric(system, qmc, replica, walker1, 1, collect(Θ+1:2Θ))
        jump_replica!(replica, 1)

        sweep!_symmetric(system, qmc, replica, walker2, 2, collect(Θ+1:2Θ))
        jump_replica!(replica, 2)

        sweep!_symmetric(system, qmc, replica, walker1, 1, collect(Θ:-1:1))
        jump_replica!(replica, 1)

        sweep!_symmetric(system, qmc, replica, walker2, 2, collect(Θ:-1:1))
        jump_replica!(replica, 2)

        return nothing
    end
end

"""
    sweep!(system, qmc, replica, walker, ridx, loop_number=1, jumpReplica=false)

    Sweep a certain walker of two copies (indexed by ridx) through the imaginary time in loop_number (by default 1) 
    times from θ to 2θ and from θ to 0
"""
function sweep!(
    system::Hubbard, qmc::QMC, 
    replica::Replica, walker::HubbardWalker, ridx::Int;
    loop_number::Int = 1, jumpReplica::Bool = false
)
    Θ = div(qmc.K,2)

    if system.useChargeHST || qmc.forceSymmetry
        for i in 1 : loop_number
            sweep!_symmetric(system, qmc, replica, walker, ridx, collect(Θ+1:2Θ))
            sweep!_symmetric(system, qmc, replica, walker, ridx, collect(Θ:-1:1))
        end

        jumpReplica && jump_replica!(replica,ridx)
        return nothing
    end
end

###################################################
##### Symmetric Sweep for Charge HS Transform #####
###################################################
function local_update!_symmetric(
    σ::AbstractArray{Int}, j::Int, l::Int, ridx::Int, 
    system::Hubbard, walker::HubbardWalker, replica::Replica;
    direction::Int = 1, forceSymmetry::Bool = false,
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
                direction=direction, forceSymmetry=forceSymmetry
            )
    saveRatio && push!(walker.tmp_r, r)
    # accept ratio
    u = useHeatbath ? real(r) / (1 + real(r)) : real(r)
    
    if rand() < u
        # accept the move, update the field and the Green's function
        walker.auxfield[j, l] *= -1
        
        ### rank-1 updates ###
        # update imaginary time G
        update_Gτ0!(Gτ0, γ, Gτ, j, ws, direction=direction)
        update_G0τ!(G0τ, γ, Gτ, j, ws, direction=direction)
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
    Θ = div(qmc.K,2)

    direction == 1 ? (
        # propagate from τ to τ+k
        Bk = system.Bk;
        Bk⁻¹ = system.Bk⁻¹;
        slice = collect(1:k);
        Bc = walker.Bc.B[cidx-Θ]
    ) : 
    (
        # propagate from τ+k to τ
        Bk = system.Bk⁻¹;
        Bk⁻¹ = system.Bk;
        slice = collect(k:-1:1);
        Bc = walker.Bc.B[cidx]
    )

    # all Green's functions
    Gτ = walker.G[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    gτ0 = walker.gτ0
    g0τ = walker.g0τ

    ws = walker.ws
    Bl = walker.Bl.B

    for i in slice
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
                                   direction=direction,
                                   forceSymmetry=qmc.forceSymmetry,
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
        wrap_Gs!(Gτ, Gτ0, G0τ, Bl[i], ws, direction=direction)
    end

    @views prod_cluster!(Bc, Bl[k:-1:1], ws.M)
    proceed_gτ0!(gτ0[cidx], Bc, Gτ, ws, direction=direction)
    proceed_g0τ!(g0τ[cidx], Bc, Gτ, ws, direction=direction)

    return nothing
end

function sweep!_symmetric(
    system::Hubbard, qmc::QMC, 
    replica::Replica, walker::HubbardWalker,
    ridx::Int, slice::Vector{Int}
)
    direction = slice[1] < slice[end] ? 1 : 2
    ### set alias ###
    Θ = div(qmc.K,2)
    Aidx = replica.Aidx
    ws = walker.ws
    logdetGA, sgnlogdetGA = replica.logdetGA, replica.sgnlogdetGA
    φ₀ = walker.φ₀[1]
    φ₀T = walker.φ₀T[1]
    # temporal factorizations
    Fτt, FτT, _ = walker.Fτ
    Fl = walker.Fl[1]
    Fr = walker.Fr[1]
    Fcl = walker.Fcl.B
    Fcr = walker.Fcr.B
    # imaginary-time-displaced Green's
    Gτ = walker.G[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    gτ0 = walker.gτ0
    g0τ = walker.g0τ

    ridx == 1 ? (G₀ = replica.G₀1; G₀′ = replica.G₀2) : 
                (G₀ = replica.G₀2; G₀′ = replica.G₀1)

    # propagate from θ to 2θ
    direction == 1 && begin
        for (i, cidx) in enumerate(slice)
            update_cluster!_symmetric(walker, replica, system, qmc, cidx, ridx, direction=1)

            # multiply the updated slice to the right factorization on the left
            Bc = walker.Bc.B[cidx-Θ]
            lmul!(Bc, Fτt, ws)

            # Gτ needs to be periodically recomputed from scratch
            mul!(FτT, Fτt, Fr, ws)
            compute_G!(walker, 1, Bl=Fcl[i], Br=FτT)

            # recompute imaginary-time-displaced Green's
            @views prod_cluster!(Gτ0, gτ0[cidx:-1:Θ+1], ws.M)
            @views prod_cluster!(G0τ, g0τ[Θ+1:cidx], ws.M)
            (cidx - Θ - 1) % 2 == 0 || @. G0τ *= -1

            # recompute G₀
            mul!(FτT, Fcl[i], Fτt, ws)
            compute_G!(G₀, φ₀, φ₀T, walker.Ul, walker.Ur, FτT, Fr)

            # recompute Grover inverse
            ridx == 1 ? begin
                    logdetGA[], sgnlogdetGA[] =  @views inv_Grover!(replica.GA⁻¹, G₀[Aidx, Aidx], G₀′[Aidx, Aidx], replica.ws)
                end :
                begin
                    logdetGA[], sgnlogdetGA[] =  @views inv_Grover!(replica.GA⁻¹, G₀′[Aidx, Aidx], G₀[Aidx, Aidx], replica.ws)
                end
            
            cidx == 2Θ && (
                copyto!(Fl, FτT); copyto!(Gτ, G₀); 
                copyto!(Gτ0, Gτ);
                copyto!(G0τ, Gτ);
                G0τ[diagind(G0τ)] .-= 1
            )
        end

        # recompute all partial factorizations
        build_propagator!(Fcl, walker.Bc, ws, K=Θ, isReverse=true, isSymmetric=true)
        # reset temporal factorizations
        ldr!(Fτt, I)
        ldr!(FτT, I)

        # copy green's function to the spin-down sector
        copyto!(walker.G[2], walker.G[1])
        qmc.forceSymmetry && conj!(walker.G[2])

        return nothing
    end

    # propagate from θ to 0
    for (i, cidx) in zip(Iterators.reverse(eachindex(slice)), slice)
        update_cluster!_symmetric(walker, replica, system, qmc, cidx, ridx, direction=2)

        # multiply the updated slice to the left factorization on the right
        Bc = walker.Bc.B[cidx]
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

        # recompute Grover inverse
        ridx == 1 ? begin
                logdetGA[], sgnlogdetGA[] =  @views inv_Grover!(replica.GA⁻¹, G₀[Aidx, Aidx], G₀′[Aidx, Aidx], replica.ws)
            end :
            begin
                logdetGA[], sgnlogdetGA[] =  @views inv_Grover!(replica.GA⁻¹, G₀′[Aidx, Aidx], G₀[Aidx, Aidx], replica.ws)
            end
        
        cidx == 1 && (
            copyto!(Fr, FτT);
            copyto!(Gτ, G₀);
            copyto!(Gτ0, Gτ);
            copyto!(G0τ, Gτ);
            G0τ[diagind(G0τ)] .-= 1
        )
    end

    # recompute all partial factorizations
    build_propagator!(Fcr, walker.Bc, ws, K=Θ, isReverse=false, isSymmetric=true)
    # reset temporal factorizations
    ldr!(Fτt, I)
    ldr!(FτT, I)

    # copy green's function to the spin-down sector
    copyto!(walker.G[2], walker.G[1])
    qmc.forceSymmetry && conj!(walker.G[2])

    return nothing
end

function jump_replica!(replica::Replica, ridx::Int)
    Aidx = replica.Aidx
    G₀ = ridx == 1 ? replica.G₀1 : replica.G₀2
    Im2GA = replica.Im2GA

    @views G′ = G₀[Aidx, Aidx]
    for i in CartesianIndices(Im2GA)
        @inbounds Im2GA[i] = -2 * G′[i]
    end
    Im2GA[diagind(Im2GA)] .+= 1 

    return replica
end
