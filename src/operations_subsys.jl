"""
    Operations for sampling in the subsystem A
    
    Walkers are updated by the weight det(I - Gₐ(θ))^-1
"""

function compute_Metropolis_ratio(
    system::System, walker::HubbardSubsysWalker,
    α::Ta, sidx::Int; direction::Int = 1
) where Ta

    # set alias
    Aidx = walker.Aidx
    G = walker.G[1]
    ImGA⁻¹ = walker.ImGA⁻¹[1]
    a = walker.a
    b = walker.b
    t = walker.t

    # direction=2 -> back propagation
    direction == 1 ? (
            Bk = system.Bk; 
            Bk⁻¹ = system.Bk⁻¹;
            G0τ = walker.G0τ[1];
            Gτ0 = walker.Gτ0[1]
        ) : 
        (
            Bk = system.Bk⁻¹; 
            Bk⁻¹ = system.Bk;
            G0τ = walker.Gτ0[1];
            Gτ0 = walker.G0τ[1]
        )

    # compute Γ = a * bᵀ
    if system.useFirstOrderTrotter  # asymmetric case
        @views mul!(a, ImGA⁻¹, G0τ[Aidx, sidx])
        @views copyto!(b, Gτ0[sidx, Aidx])
        
    else                            # symmetric case
        @views mul!(t, Bk⁻¹[Aidx, :], G0τ[:, sidx])
        @views mul!(a, ImGA⁻¹, t)

        @views transpose_mul!(b, Gτ0[sidx, :], Bk[:, Aidx])
    end

    d = α * (1 - G[sidx, sidx])
    γ = α / (1 + d)

    ρ = γ / (1 - γ*dot(a,b))
    # accept ratio
    r = 1 / (1 - γ*dot(a,b))^2

    return r, γ, ρ
end

function update_invImGA!(walker::HubbardSubsysWalker, ρ::T) where T
    ImGA⁻¹ = walker.ImGA⁻¹[1]
    a = walker.a
    b = walker.b
    bᵀ = walker.t
    dImGA⁻¹ = walker.wsA.M

    transpose_mul!(bᵀ, b, ImGA⁻¹)

    kron!(dImGA⁻¹, ρ, a, bᵀ)
    @. ImGA⁻¹ += dImGA⁻¹
end

"""
    compute_invImGA

    Given GA, compute (I - GA)⁻¹ in a numerically stable fashion
"""
function compute_invImGA!(ImGA⁻¹::AbstractMatrix, GA::AbstractMatrix, wsA::LDRWorkspace)
    
    # compute (I - GA) in-place
    ImGA = ImGA⁻¹
    @inbounds for i in eachindex(GA)
        ImGA[i] = -GA[i]
    end
    ImGA[diagind(ImGA)] .+= 1

    # take inverse
    inv_lu!(ImGA⁻¹, wsA.lu_ws)

    return ImGA⁻¹
end
