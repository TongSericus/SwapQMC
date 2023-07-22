"""
    One- and Two-body Density Matrices for Observable Measurements
"""

using SwapQMC

struct DensityMatrix{T}
    # One-body RDM: <c_i^{+} cj>
    # In ground state, (ρ₁)ᵢⱼ = I - Gⱼᵢ
    ρ₁::Matrix{T}
end

function DensityMatrix(G::AbstractMatrix)

    ρ₁ = I - transpose(G)

    return DensityMatrix{eltype(ρ₁)}(ρ₁)
end

function update!(
    ρ::DensityMatrix, walker::HubbardWalker, spin::Int
)
    ρ₁= ρ.ρ₁
    G = walker.G[spin]
    transpose!(ρ₁, G)

    @inbounds for i in eachindex(ρ₁)
        ρ₁[i] *= -1
    end
    @views ρ₁[diagind(ρ₁)] .+= 1

    return nothing
end

"""
    ρ₂(ρ::DensityMatrix, i::Int, j::Int, k::Int, l::Int)

    Compute the two-body estimator <cᵢ⁺ cⱼ cₖ⁺ cₗ> using Wick's theorem as
    <cᵢ⁺ cⱼ cₖ⁺ cₗ> = <cᵢ⁺ cⱼ> <cₖ⁺ cₗ> + <cᵢ⁺ cl> (δₖⱼ - <cₖ⁺ cⱼ>)
"""
function ρ₂(ρ::DensityMatrix, i::Int, j::Int, k::Int, l::Int)
    ρ₁ = ρ.ρ₁
    return ρ₁[i, j] * ρ₁[k, l] + ρ₁[i, l] * ((k==j) - ρ₁[k, j])
end
