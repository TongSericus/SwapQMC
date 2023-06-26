function compute_Metropolis_ratio(
    system::System, walker::HubbardSubsysWalker,
    spin::Int, α::Ta, sidx::Int;
    direction::Int = 1
) where {W, Ta}

    # set alias
    Aidx = walker.Aidx
    ImGA⁻¹ = walker.ImGA⁻¹
    a = walker.a
    b = walker.b
    t = walker.t

    # compute I - G
    ImG = walker.ws.M
    G = walker.G[spin]
    @inbounds for i in eachindex(G)
        ImG[i] = -G[i]
    end
    ImG[diagind(ImG)] .+= 1

    # direction=2 -> back propagation
    direction == 1 ? (Bk = system.Bk; Bk⁻¹ = system.Bk⁻¹) : (Bk = system.Bk⁻¹; Bk⁻¹ = system.Bk)

    # compute Γ = a * bᵀ
    if system.useFirstOrderTrotter  # asymmetric case
        @views mul!(a, ImGA⁻¹, ImG[Aidx, sidx])
        @views copyto!(b, G[sidx, Aidx])
        
    else                            # symmetric case
        @views mul!(t, ImGA⁻¹, Bk⁻¹[Aidx, :])
        @views mul!(a, t, ImG[:, sidx])

        @views transpose_mul!(b, G[sidx, :], Bk[:, Aidx])
    end

    d = α * (1 - G[sidx, sidx])
    γ = α / (1 + d)

    ρ = γ / (1 + γ*dot(a,b))
    # accept ratio
    r = 1 / (1 + γ*dot(a,b))^2

    return r, γ, ρ
end

function update_invImGA!(walker::HubbardSubsysWalker, ρ::T) where T
    ImGA⁻¹ = walker.ImGA⁻¹
    a = walker.a
    b = walker.b
    bᵀ = walker.t
    dImGA⁻¹ = walker.wsA.M

    transpose_mul!(bᵀ, b, ImGA⁻¹)

    kron!(dImGA⁻¹, ρ, a, bᵀ)
    @. ImGA⁻¹ -= dImGA⁻¹
end

"""
    compute_invImGA

    Given GA, compute (I - GA)⁻¹ in a numerically stable fashion
"""
function compute_invImGA!(ImGA⁻¹::AbstractMatrix, GA::AbstractMatrix, wsA::LDRWorkspace)
    
    # compute (I - GA) in-place
    ImGA = ImGA⁻¹
    @inbounds for i in eachindex(GA₁)
        ImGA[i] = -GA[i]
    end
    ImGA[diagind(ImGA)] .+= 1

    # take inverse
    inv_lu!(ImGA⁻¹, wsA.lu_ws)

    return ImGA⁻¹
end
