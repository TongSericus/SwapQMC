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

    for i in eachindex(ρ₁)
        @inbounds ρ₁[i] *= -1
    end
    @views ρ₁[diagind(ρ₁)] .+= 1 

    return nothing
end

"""
    Compute the two-body estimator <cᵢ⁺ cⱼ cₖ⁺ cₗ> using Wick's theorem
"""
function ρ₂(ρ::DensityMatrix, i::Int, j::Int, k::Int, l::Int)
    ρ₁ = ρ.ρ₁
    return ρ₁[j, i] * ρ₁[l, k] + ρ₁[l, i] * ((k==j) - ρ₁[j, k])
end
