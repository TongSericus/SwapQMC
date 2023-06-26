
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
    Ul = walker.Ul
    Ur = walker.Ur

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

function proceed_gτ0!(
    gτ0::AbstractMatrix, Bτ::AbstractMatrix, Gτ::AbstractMatrix, ws::LDRWorkspace; 
    direction::Int = 1
)
    direction == 1 && begin
        mul!(gτ0, Gτ, Bτ)
        
        return nothing
    end

    mul!(gτ0, Bτ, Gτ)

    return nothing
end

function proceed_g0τ!(
    g0τ::AbstractMatrix, Bτ::AbstractMatrix, Gτ::AbstractMatrix, ws::LDRWorkspace; 
    direction::Int = 1
)
    # compute B⁻¹
    Bτ⁻¹ = ws.M′
    copyto!(Bτ⁻¹, Bτ)
    inv_lu!(Bτ⁻¹, ws.lu_ws)

    direction == 1 && begin
        # compute B⁻¹
        Bτ⁻¹ = ws.M′
        copyto!(Bτ⁻¹, Bτ)
        inv_lu!(Bτ⁻¹, ws.lu_ws)
        
        mul!(g0τ, Bτ⁻¹, Gτ - I)

        return nothing
    end

    # compute B⁻¹
    Bτ⁻¹ = ws.M′
    copyto!(Bτ⁻¹, Bτ)
    inv_lu!(Bτ⁻¹, ws.lu_ws)

    mul!(g0τ, Gτ - I, Bτ⁻¹)

    return nothing
end