
"""
    compute_G!(G, φ₀, φ₀ᵀ, Ul, Ur, Bl, Br)

    compute the ground state Green's function from scratch
"""
function compute_G!(
    G::AbstractMatrix, φ₀::AbstractMatrix, φ₀ᵀ::AbstractMatrix,
    Ul::AbstractMatrix, Ur::AbstractMatrix,
    Bl::LDR{T,E}, Br::LDR{T,E}
) where {T,E}
    lmul!_svd(Ul, φ₀ᵀ, Bl)
    rmul!_svd(Ur, Br, φ₀)
    mul!(G, Ur*inv(Ul*Ur), Ul)

    # compute G <- I - G
    for i in CartesianIndices(G)
        @inbounds G[i] = -G[i]
    end
    G[diagind(G)] .+= 1

    return G
end

"""
    compute_G!(walker, spin)
    
    compute the ground state Green's function given the walker
"""
function compute_G!(
    walker::GSWalker, spin::Int; 
    Bl::LDR = walker.Fl[spin], Br::LDR = walker.Fr[spin]
)
    G = walker.G[spin]
    # current LDR decomposition can't deal with non-square matrix
    φ₀ = walker.φ₀[spin]
    φ₀ᵀ= walker.φ₀T[spin]
    Ul = walker.Ul[spin]
    Ur = walker.Ur[spin]

    lmul!_svd(Ul, φ₀ᵀ, Bl)
    rmul!_svd(Ur, Br, φ₀)
    mul!(G, Ur*inv(Ul*Ur), Ul)

    # compute G <- I - G
    for i in CartesianIndices(G)
        @inbounds G[i] = -G[i]
    end
    G[diagind(G)] .+= 1 

    return G
end
