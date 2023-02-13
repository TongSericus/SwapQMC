### Basic operations ###
"""
    compute_Metropolis_ratio(G, α, i, sidx)

    Compute the ratio of determinants using Green's function
    det(M_new) / det(M_old) = 1 + α * (1 - G[i, i])
    if only the ith auxiliary field is flipped
"""
function compute_Metropolis_ratio(
    G::AbstractArray{T}, α::Ta, i::Int, sidx::Int
) where {T, Ta}
    d_up = 1 + α[1, i] * (1 - G[1][sidx, sidx])
    d_dn = 1 + α[2, i] * (1 - G[2][sidx, sidx])
    r = abs(d_up * d_dn)

    return r, d_up, d_dn
end

"""
    Compute the ratio of determinants using Green's function, with
    the auxiliary field being complex and spin-up and spin-down channel
    are identical (hence only a single G is required)
"""
function compute_Metropolis_ratio(
    G::AbstractMatrix, α::Ta, i::Int, sidx::Int
) where Ta
    γ = α[1, i]
    d = γ * (1 - G[sidx, sidx])
    # accept ratio
    r = (1 + d)^2 / (γ + 1)

    return real(r), d+1
end

"""
    kron!(C, α, a, b)

    In-place operation of C = α * u * wᵀ
"""
function Base.kron!(C::AbstractMatrix, α::Number, a::AbstractVector, b::AbstractVector)
    for i in eachindex(a)
        for j in eachindex(a)
            @inbounds C[j, i] = α * a[j] * b[i]
        end
    end
end

"""
    update_G!(G, α, d, sidx, ws)

    Fast update the Green's function at the ith site:
    G ← G - α_{i, σ} / d_{i, i} * u * wᵀ
    where u = (I - G)e₁, w = Gᵀe₁
"""
function update_G!(G::AbstractMatrix{T}, α::Ta, d::Ta, sidx::Int64, ws::LDRWorkspace{T, E}) where {T, Ta, E}
    ImG = ws.M
    dG = ws.M′
    g = @view G[sidx, :]
    Img = @view ImG[:, sidx]

    # compute I - G
    for i in eachindex(g)
        @inbounds Img[i] = -G[i, sidx]
    end
    Img[sidx] += 1

    # compute (I - G)eᵢ * Gᵀeᵢ
    @views kron!(dG, α / d, Img, g)

    G .-= dG
end

"""
    wrap_G!(G, B, ws)

    Compute G ← B * G * B⁻¹
"""
function wrap_G!(G::AbstractMatrix{T}, B::AbstractMatrix{TB}, ws::LDRWorkspace{T, E}) where {T, TB, E}
    mul!(ws.M, B, G)
    
    B⁻¹ = ws.M′
    copyto!(B⁻¹, B)
    inv_lu!(B⁻¹, ws.lu_ws)
    mul!(G, ws.M, B⁻¹)
end

function wrap_G!(G::AbstractMatrix{T}, B::AbstractMatrix{TB}, B⁻¹::AbstractMatrix{TB}, ws::LDRWorkspace{T, E}) where {T, TB, E}
    mul!(ws.M, B, G)
    mul!(G, ws.M, B⁻¹)
end

flip_HSField(σ::Int) = Int64((σ + 1) / 2 + 1)

"""
    Propagate the full space-time lattice
"""
function run_full_propagation(
    auxfield::AbstractMatrix{Int64}, system::System, qmc::QMC, ws::LDRWorkspace{T,E}; 
    isReverse::Bool = true
) where {T, E}
    K = qmc.K 
    V = system.V
    si = qmc.stab_interval
    K_interval = qmc.K_interval

    # initialize partial matrix products
    Tb = eltype(system.auxfield)
    B = [Matrix{Tb}(I, V, V), Matrix{Tb}(I, V, V)]
    MatProd = Cluster(V, 2 * K, T = Tb)
    F = ldrs(B[1], 2)
    FC = Cluster(B = ldrs(B[1], 2 * K))

    Bm = MatProd.B
    Bf = FC.B

    isReverse && begin
        for i in K:-1:1
            for j = 1 : K_interval[i]
            @views σ = auxfield[:, (i - 1) * si + j]
            imagtime_propagator!(B[1], B[2], σ, system, tmpmat = ws.M)
            Bm[i] = B[1] * Bm[i]            # spin-up
            Bm[K + i] = B[2] * Bm[K + i]    # spin-down
        end

        # save all partial products
        copyto!(Bf[i], F[1])
        copyto!(Bf[K + i], F[2])

        rmul!(F[1], Bm[i], ws)
        rmul!(F[2], Bm[K + i], ws)
    end

        return F, MatProd, FC
    end

    for i in 1:K
        for j = 1 : K_interval[i]
            @views σ = auxfield[:, (i - 1) * si + j]
            imagtime_propagator!(B[1], B[2], σ, system, tmpmat = ws.M)
            Bm[i] = B[1] * Bm[i]            # spin-up
            Bm[K + i] = B[2] * Bm[K + i]    # spin-down
        end

        copyto!(Bf[i], F[1])
        copyto!(Bf[K + i], F[2])

        lmul!(Bm[i], F[1], ws)
        lmul!(Bm[K + i], F[2], ws)
    end

    return F, MatProd, FC
end

"""
    Propagate the full space-time lattice given the matrix clusters
"""
function run_full_propagation(
    MatProd::Cluster{C}, ws::LDRWorkspace{T,E};
    isReverse::Bool = true,
    K = div(length(MatProd.B), 2),
    V = size(MatProd.B[1]),
    i = eltype(ws.M) <: Real ? 1.0 : 1.0+0.0im,
    F = ldrs(Matrix(i*I, V), 2),
    FC = Cluster(B = ldrs(Matrix(i*I, V), 2 * K))
) where {C, T, E}

    Bm = MatProd.B
    Bf = FC.B

    isReverse && begin 
        for i in K:-1:1
            copyto!(Bf[i], F[1])
            copyto!(Bf[K + i], F[2])

            rmul!(F[1], Bm[i], ws)
            rmul!(F[2], Bm[K + i], ws)
        end

        return F
    end

    for i in 1:K
        copyto!(Bf[i], F[1])
        copyto!(Bf[K + i], F[2])

        lmul!(Bm[i], F[1], ws)
        lmul!(Bm[K + i], F[2], ws)
    end
    
    return F
end

function run_full_propagation_oneside(
    MatProd::Cluster{C}, ws::LDRWorkspace{T,E};
    isReverse::Bool = true,
    K = div(length(MatProd.B), 2),
    V = size(MatProd.B[1]),
    i = eltype(ws.M) <: Real ? 1.0 : 1.0+0.0im,
    F = ldr(Matrix(i*I, V)),
    FC = Cluster(B = ldrs(Matrix(i*I, V), K))
) where {C, T, E}

    Bm = MatProd.B
    Bf = FC.B

    isReverse && begin 
        for i in K:-1:1
            copyto!(Bf[i], F)

            rmul!(F, Bm[i], ws)
        end

        return F
    end

    for i in 1:K
        copyto!(Bf[i], F)

        lmul!(Bm[i], F, ws)
    end
    
    return F
end