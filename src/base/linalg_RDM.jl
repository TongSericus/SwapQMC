"""
    Extra Linear Algebra Operations
"""

################################################################
##### Additional Functions for StableLinearAlgebra package #####
################################################################

# iteration for destructuring into components
Base.iterate(S::LDR) = (S.L, Val(:d))
Base.iterate(S::LDR, ::Val{:d}) = (S.d, Val(:R))
Base.iterate(S::LDR, ::Val{:R}) = (S.R, Val(:done))
Base.iterate(S::LDR, ::Val{:done}) = nothing

Base.similar(S::LDR{T, E}) where {T, E} = ldr(S)

# Diagonalization
LinearAlgebra.eigvals(F::LDR{T, E}) where {T, E} = eigvals(Diagonal(F.d) * F.R * F.L, sortby = abs)

#############################################
##### Functions for non-square matrices #####
#############################################
function rmul_qr!(Ur::AbstractMatrix, U::LDR{T,E}, V::AbstractMatrix) where {T,E}

    mul!(Ur, U.R, V)
    lmul_D!(U.d, Ur)
    Q,_ = qr!(Ur)
    mul!(Ur, U.L, Matrix(Q))

    return Ur
end

function lmul_qr!(Ul::AbstractMatrix, U::AbstractMatrix, V::LDR{T,E}) where {T,E}
    
    mul!(Ul, U, V.L)
    rmul_D!(Ul, V.d)
    _,R = qr!(Ul)
    for i in axes(R,2)
        @views R[:, i] ./= norm(R[:, i])
    end
    mul!(Ul, R, V.R)

    return Ul
end
########################################################
##### Matrix Operations for Reduced Density Matrix #####
########################################################

function det_UpV(U::LDR{T,E}, V::LDR{T,E}, ws::LDRWorkspace{T,E}) where {T,E}
    """
        det_UpV(U::LDR, V::LDR, ws::LDRWorkspace)

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

"""
    invAmI!(A)

    Stable calculation of A⁻¹ - I via SVD
"""
function invAmI!(A::AbstractMatrix{T}) where T
    # SVD with QR iteration for better accuracy
    A_svd = svd!(A, alg = LinearAlgebra.QRIteration())
    U, d, V = A_svd
    Uᵀ, Vᵀ = U', V'

    # calculate VᵀU
    VᵀU = Vᵀ * U

    # calculate VᵀU <- d⁻¹ - VᵀU
    @. VᵀU *= -1
    @inbounds for i in eachindex(d)
        VᵀU[i, i] += 1/d[i]
    end

    return V, VᵀU, Uᵀ
end

function compute_etgHam(GA::AbstractMatrix{T}) where T
    # compute GA⁻¹ - I
    U, D, V = invAmI!(GA)
    F = svd!(D, alg = LinearAlgebra.QRIteration())

    return U*F.U, F.S, F.Vt*V
end

function compute_etgHam(
    G₁::AbstractMatrix{T}, G₂::AbstractMatrix{T}, 
    Aidx::Vector{Int}, ws::LDRWorkspace{T,E}
) where {T,E}

    # compute GA₁⁻¹ - I
    GA₁ = ws.M
    @views copyto!(GA₁, G₁[Aidx, Aidx])
    U₁, d₁, V₁ = invAmI!(GA₁)
    # compute GA₂⁻¹ - I
    GA₂ = ws.M
    @views copyto!(GA₂, G₂[Aidx, Aidx])
    U₂, d₂, V₂ = invAmI!(GA₂)

    # merge (GA₂⁻¹ - I)(GA₁⁻¹ - I)
    U₁d₁ = ws.M
    mul!(U₁d₁, U₁, d₁)
    V₂U₁d₁ = ws.M′
    mul!(V₂U₁d₁, V₂, ws.M)
    d₂V₂U₁d₁ = ws.M
    mul!(d₂V₂U₁d₁, d₂, V₂U₁d₁)
    F = svd!(d₂V₂U₁d₁, alg = LinearAlgebra.QRIteration())

    return U₂*F.U, F.S, F.Vt*V₁
end

"""
    ImA!(G, A, ws)
        
    Stable calculation of G = I - A
"""
function ImA!(G::LDR{T,E}, A::LDR{T,E}, ws::LDRWorkspace{T,E}) where {T,E}
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
    @inbounds for i in eachindex(dₐ)
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

################################################
##### Matrix Operations for Swap Algorithm #####
################################################

reset!(U::AbstractMatrix) = let
    @. U = 0.0
end

reset!(U::LDR{T,E}) where {T, E} = let
    @. U.L = 0.0
    @. U.d = 1.0
    @. U.R = 0.0
end

function expand!(U::AbstractMatrix{T}, V::AbstractMatrix{T}, LA::Int, LB::Int, ridx::Int) where T
    """
        expand!(U, V, LA, LB, ridx)

        Fill the matrix/decomposition U by expanding the matrix/decomposition V via inserting a unit matrix
    """
    reset!(U)
    L = LA + LB

    if ridx == 1
        @views copyto!(U[1:L, 1:L], V[1:L, 1:L])
        U_sub = @view U[L+1:end, L+1:end]
        U_sub[diagind(U_sub)] .= 1

    elseif ridx == 2
        @views copyto!(U[1:LA, 1:LA], V[1:LA, 1:LA])
        @views copyto!(U[L+1:end, L+1:end], V[LA+1:end, LA+1:end])

        @views copyto!(U[1:LA, L+1:end], V[1:LA, LA+1:end])
        @views copyto!(U[L+1:end, 1:LA], V[LA+1:end, 1:LA])

        U_sub = @view U[LA+1:L, LA+1:L]
        U_sub[diagind(U_sub)] .= 1
    end

    return nothing
end

function expand!(
    U::AbstractVector{T}, V::AbstractVector{T}, 
    LA::Int, LB::Int, ridx::Int
) where {T<:AbstractMatrix}
    length(U) == length(V) || @error "Two vectors of matrix must have same length"

    for i in eachindex(U)
        expand!(U[i], V[i], LA, LB, ridx)
    end

    return nothing
end

function expand!(U::LDR{T,E}, V::LDR{T,E}, ridx::Int) where {T, E}
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
    ridx == 1 ? (idx = Lv) : (idx = Lv - δL)

    # find the last diagonal value that is larger than 1
    lᵥ = findfirst(x -> x < 1, dᵥ)
    lᵥ === nothing ? lᵥ = length(dᵥ) : lᵥ -= 1

    reset!(U)

    # fill d
    dᵥidx = vcat(1:lᵥ, lᵥ+1+δL:Lu)
    @views copyto!(dᵤ[dᵥidx], dᵥ)

    # fill L
    @views copyto!(Lᵤ[1:idx, 1:lᵥ], Lᵥ[1:idx, 1:lᵥ])
    @views copyto!(Lᵤ[1:idx, lᵥ+1+δL:Lu], Lᵥ[1:idx, lᵥ+1:Lv])
    idx == Lv || (@views copyto!(Lᵤ[idx+δL+1:Lu, 1:lᵥ], Lᵥ[idx+1:Lv, 1:lᵥ]); 
                  @views copyto!(Lᵤ[idx+δL+1:Lu, lᵥ+1+δL:Lu], Lᵥ[idx+1:Lv, lᵥ+1:Lv])
                )
    Lᵤ_sub = @view Lᵤ[idx+1 : idx+δL, lᵥ+1 : lᵥ+δL]
    Lᵤ_sub[diagind(Lᵤ_sub)] .= 1

    # fill R
    @views copyto!(Rᵤ[1:lᵥ, 1:idx], Rᵥ[1:lᵥ, 1:idx])
    @views copyto!(Rᵤ[lᵥ+1+δL:Lu, 1:idx], Rᵥ[lᵥ+1:Lv, 1:idx])
    idx == Lv || (@views copyto!(Rᵤ[1:lᵥ, idx+δL+1:Lu], Rᵥ[1:lᵥ, idx+1:Lv]); 
                  @views copyto!(Rᵤ[lᵥ+1+δL:Lu, idx+δL+1:Lu], Rᵥ[lᵥ+1:Lv, idx+1:Lv])
                )
    Rᵤ_sub = @view Rᵤ[lᵥ+1 : lᵥ+δL, idx+1 : idx+δL]
    Rᵤ_sub[diagind(Rᵤ_sub)] .= 1

    return nothing
end

### New Functions ###
function transpose_mul!(C::AbstractVector, A::AbstractVector, B::AbstractMatrix)
    """
        Compute C = Aᵀ * B
    """
    @inbounds @fastmath for i in eachindex(C)
        Ci = zero(eltype(C))
        for j in eachindex(A)
            Ci += A[j] * B[j, i]
        end
        C[i] = Ci
    end
end

function inv_Grover!(
    GA⁻¹::AbstractMatrix{T},
    GA₁::AbstractMatrix{T}, GA₂::AbstractMatrix{T}, 
    ws::LDRWorkspace{T, E}
) where {T, E}
    """
        inv_Grover!(GA⁻¹, GA₁, GA₂, ws)

        Compute the Grover inverse GA⁻¹ = [GA1 * GA2 + (I - GA1) * (I - GA2)]⁻¹
    """
    mul!(GA⁻¹, GA₁, GA₂)

    ImGA₁ = ws.M′
    ImGA₂ = ws.M″
    @inbounds for i in eachindex(GA₁)
        ImGA₁[i] = -GA₁[i]
        ImGA₂[i] = -GA₂[i]
    end
    ImGA₁[diagind(ImGA₁)] .+= 1
    ImGA₂[diagind(ImGA₂)] .+= 1

    ImG = ws.M
    mul!(ImG, ImGA₁, ImGA₂)

    @. GA⁻¹ += ImG
    inv_lu!(GA⁻¹, ws.lu_ws)
end
