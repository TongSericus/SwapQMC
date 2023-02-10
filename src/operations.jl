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
    F = ldrs(Matrix(1.0I, V), 2),
    FC = Cluster(B = ldrs(Matrix(1.0I, V), 2 * K))
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
