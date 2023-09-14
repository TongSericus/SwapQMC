"""
    Linear Algebra operations used in the ground state propagation
"""

function rmul_D!(M::AbstractMatrix, d::AbstractVector)

    @inbounds @fastmath for c in eachindex(d)
        for r in axes(M,1)
            M[r,c] *= d[c]
        end
    end

    return nothing
end

function lmul_D!(d::AbstractVector, M::AbstractMatrix)

    @inbounds @fastmath for c in axes(M,2)
        for r in eachindex(d)
            M[r,c] *= d[r]
        end
    end

    return nothing
end

function lmul!_svd(Ul::AbstractMatrix, U::AbstractMatrix, V::LDR{T,E}) where {T,E}
    
    mul!(Ul, U, V.L)
    rmul_D!(Ul, V.d)
    L,D,R = svd!(Ul, alg = LinearAlgebra.QRIteration())
    mul!(Ul, R', V.R)

    return L, D, Ul
end

function rmul!_svd(Ur::AbstractMatrix, U::LDR{T,E}, V::AbstractMatrix) where {T,E}

    mul!(Ur, U.R, V)
    lmul_D!(U.d, Ur)
    L,D,R = svd!(Ur, alg = LinearAlgebra.QRIteration())
    mul!(Ur, U.L, L)

    return Ur, D, R
end
