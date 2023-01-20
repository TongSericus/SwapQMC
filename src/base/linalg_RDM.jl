"""
    Matrix Operations for Reduced Density Matrix
"""

import StableLinearAlgebra as Slinalg

function det_UpV(U::LDR{T,E}, V::LDR{T,E}, ws::LDRWorkspace{T,E}) where {T,E}
    """
        Compute the numerically stable determinant G=det(U+V), where U and V are represented by factorizations,
        using the formula
        G = det(U + V) = det(Lᵤ * Dᵤ₊ * M * Dᵥ₊ * Rᵥ) with |det(Lᵤ)| = |det(Rᵥ)| = 1
    """
    Lᵤ = U.L
    dᵤ = U.d
    Rᵤ = U.R
    Lᵥ = V.L
    dᵥ = V.d
    Rᵥ = V.R

    # calculate sign(det(Lᵤ)) and log(|det(Lᵤ)|)
    copyto!(ws.M, Lᵤ)
    logdetLᵤ, sgndetLᵤ = det_lu!(ws.M, ws.lu_ws)

    # calculate Rᵥ⁻¹
    Rᵥ⁻¹ = ws.M′
    copyto!(Rᵥ⁻¹, Rᵥ)
    logdetRᵥ⁻¹, sgndetRᵥ⁻¹ = Slinalg.inv_lu!(Rᵥ⁻¹, ws.lu_ws)

    # calcuate Dᵥ₊ = max(Dᵥ, 1)
    dᵥ₊ = ws.v
    @. dᵥ₊ = max(dᵥ, 1)

    # calculate sign(det(Dᵥ₊)) and log(|det(Dᵥ₊)|)
    logdetDᵥ₊, sgndetDᵥ₊ = Slinalg.det_D(dᵥ₊)

    # calculate Rᵥ⁻¹⋅Dᵥ₊⁻¹
    Slinalg.rdiv_D!(Rᵥ⁻¹, dᵥ₊)
    Rᵥ⁻¹Dᵥ₊⁻¹ = Rᵥ⁻¹

    # calcuate Dᵤ₊ = max(Dᵤ, 1)
    dᵤ₊ = ws.v
    @. dᵤ₊ = max(dᵤ, 1)

    # calculate sign(det(Dᵥ₊)) and log(|det(Dᵥ₊)|)
    logdetDᵤ₊, sgndetDᵤ₊ = Slinalg.det_D(dᵤ₊)
    
    # calcualte Dᵤ₊⁻¹⋅Lᵤᵀ
    adjoint!(ws.M, Lᵤ)
    Slinalg.ldiv_D!(dᵤ₊, ws.M)
    Dᵤ₊⁻¹Lᵤᵀ = ws.M

    # calculate Dᵤ₋ = min(Dᵤ, 1)
    dᵤ₋ = ws.v
    @. dᵤ₋ = min(dᵤ, 1)
    
    # calculate Dᵤ₋⋅Rᵤ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹
    mul!(ws.M″, Rᵤ, Rᵥ⁻¹Dᵥ₊⁻¹)      # Rᵤ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹
    Slinalg.lmul_D!(dᵤ₋, ws.M″)     # Dᵤ₋⋅[Rᵤ⋅Rᵥ⁻¹⋅Dᵥ₊]

    # calculate Dᵥ₋ = min(Dᵥ, 1)
    dᵥ₋ = ws.v
    @. dᵥ₋ = min(dᵥ, 1)

    # calculate Dᵤ₊⁻¹⋅Lᵤᵀ⋅Lᵥ⋅Dᵥ₋
    mul!(ws.M′, Dᵤ₊⁻¹Lᵤᵀ, Lᵥ)
    Slinalg.rmul_D!(ws.M′, dᵥ₋)

    # calculate M = Dᵤ₋⋅Rᵤ⋅Rᵥ⁻¹⋅Dᵥ₊ + Dᵤ₊⁻¹⋅Lᵤᵀ⋅Lᵥ⋅Dᵥ₋
    M = ws.M″
    BLAS.axpy!(1.0, ws.M′, M)
    logdetM, sgndetM = Slinalg.det_lu!(M, ws.lu_ws)

    # calculate sign(det(U+V)) and log(|det(U+V)|)
    sgndetUpV = conj(sgndetRᵥ⁻¹) * sgndetDᵥ₊ * sgndetM * sgndetDᵤ₊ * sgndetLᵤ
    logdetUpV = logdetDᵥ₊ + logdetM + logdetDᵤ₊

    return real(logdetUpV), sgndetUpV
end

function ImA!(G::LDR{T,E}, A::LDR{T,E}, ws::LDRWorkspace{T,E}) where {T,E}
    """
        Compute G=I-A as a factorization stablely
    """
    Lₐ = A.L
    dₐ = A.d
    Rₐ = A.R

    # calculate Rₐ⁻¹
    Rₐ⁻¹ = ws.M′
    copyto!(Rₐ⁻¹, Rₐ)
    Slinalg.inv_lu!(Rₐ⁻¹, ws.lu_ws)

    # calculate Lₐ†Rₐ⁻¹
    Lₐ⁻¹Rₐ⁻¹ = Lₐ' * Rₐ⁻¹

    # calculate L†Rₐ⁻¹ - D
    @inbounds for i in 1 : length(dₐ)
        Lₐ⁻¹Rₐ⁻¹[i, i] -= dₐ[i]
    end

    # calculate G = L′D′R′
    ldr!(G, Lₐ⁻¹Rₐ⁻¹, ws)

    # calculate L * L′
    L′ = ws.M′
    copyto!(L′, G.L)
    mul!(G.L, Lₐ, L′)

    # calculate R′ * R
    R′ = ws.M′
    copyto!(R′, G.R)
    mul!(G.R, R′, Rₐ)

    return nothing
end

compute_HA!(HA::LDR{T,E}, ImGA::LDR{T,E}, ws::LDRWorkspace{T,E}) where {T, E} = rdiv!(HA, ImGA, ws)

merge_HA!(HA::LDR{T,E}, HA′::LDR{T,E}, ws::LDRWorkspace{T,E})  where {T, E} = rmul!(HA, HA′, ws)

"""
    Matrix Operations for Swap Algorithm
"""
function reset!(U::LDR{T,E}) where {T, E}
    @. U.L = 0.0
    @. U.d = 1.0
    @. U.R = 0.0
end

function expand!(U::LDR{T,E}, V::LDR{T,E}, idx::Int) where {T, E}
    """
        Fill U by expanding V via inserting a unit matrix starting at the position (idx+1, idx+1)
    """
    Lᵤ = U.L
    dᵤ = U.d
    Rᵤ = U.R
    Lᵥ = V.L
    dᵥ = V.d
    Rᵥ = V.R

    Lu = length(U.d)
    Lv = length(V.d)

    δL = Lu - Lv
    δL > 0 || @error "Dimension of U must be larger than dimension of V"

    # find the last diagonal value that is larger than 1
    lᵥ = 0
    while V.d[lᵥ+1] > 1
        lᵥ += 1
    end

    reset!(U)

    # fill d
    dᵤ[vcat(1:lᵥ, lᵥ+1+δL:Lu)] .= dᵥ

    # fill L
    @views Lᵤ[1:idx, 1:lᵥ] .= Lᵥ[1:idx, 1:lᵥ]
    @views Lᵤ[1:idx, lᵥ+1+δL:Lu] .= Lᵥ[1:idx, lᵥ+1:Lv]
    idx == Lv || (@views Lᵤ[idx+δL+1:Lu, 1:lᵥ] .= Lᵥ[idx+1:Lv, 1:lᵥ]; @views Lᵤ[idx+δL+1:Lu, lᵥ+1+δL:Lu] .= Lᵥ[idx+1:Lv, lᵥ+1:Lv])
    Lᵤ_sub = @view Lᵤ[idx+1 : idx+δL, lᵥ+1 : lᵥ+δL]
    Lᵤ_sub[diagind(Lᵤ_sub)] .= 1

    # fill R
    @views Rᵤ[1:lᵥ, 1:idx] .= Rᵥ[1:lᵥ, 1:idx]
    @views Rᵤ[lᵥ+1+δL:Lu, 1:idx] .= Rᵥ[lᵥ+1:Lv, 1:idx]
    idx == Lv || (@views Rᵤ[1:lᵥ, idx+δL+1:Lu] .= Rᵥ[1:lᵥ, idx+1:Lv]; @views Rᵤ[lᵥ+1+δL:Lu, idx+δL+1:Lu] .= Rᵥ[lᵥ+1:Lv, idx+1:Lv])
    Rᵤ_sub = @view Rᵤ[lᵥ+1 : lᵥ+δL, idx+1 : idx+δL]
    Rᵤ_sub[diagind(Rᵤ_sub)] .= 1

    return nothing
end
