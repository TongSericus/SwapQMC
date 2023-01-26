"""
   Operations for extended systems
"""

function run_full_propagation(
    auxfield::AbstractMatrix{Int64}, system::System, qmc::QMC, ws::LDRWorkspace{T,E}; 
    K = qmc.K, stab_interval = qmc.stab_interval, 
    K_interval = qmc.K_interval,
    V = system.V,
    B = [Matrix{Float64}(I, V, V), Matrix{Float64}(I, V, V)],
    FC = Cluster(B = ldrs(B[1], 2 * K))
) where {T, E}
    """
        Propagate the full space-time lattice
    """
    MP = Cluster(V, 2 * K)

    F = ldrs(B[1], 2)

    for i in 1 : K

        for j = 1 : K_interval[i]
            @views σ = auxfield[:, (i - 1) * stab_interval + j]
            singlestep_matrix!(B[1], B[2], σ, system, tmpmat = ws.M)
            MP.B[i] = B[1] * MP.B[i]            # spin-up
            MP.B[K + i] = B[2] * MP.B[K + i]    # spin-down
        end

        copyto!(FC.B[i], F[1])
        copyto!(FC.B[K + i], F[2])

        lmul!(MP.B[i], F[1], ws)
        lmul!(MP.B[K + i], F[2], ws)
    end

    return F, MP, FC
end

function run_full_propagation_reverse(
    auxfield::AbstractMatrix{Int64}, system::System, qmc::QMC, ws::LDRWorkspace{T,E}; 
    K = qmc.K, stab_interval = qmc.stab_interval, 
    K_interval = qmc.K_interval,
    V = system.V,
    B = [Matrix{Float64}(I, V, V), Matrix{Float64}(I, V, V)],
    FC = Cluster(B = ldrs(B[1], 2 * K))
) where {T, E}
    """
        Propagate the full space-time lattice in the reverse order
    """
    MP = Cluster(V, 2 * K)

    F = ldrs(B[1], 2)

    for i in K : -1 : 1

        for j = 1 : K_interval[i]
            @views σ = auxfield[:, (i - 1) * stab_interval + j]
            singlestep_matrix!(B[1], B[2], σ, system, tmpmat = ws.M)
            MP.B[i] = B[1] * MP.B[i]            # spin-up
            MP.B[K + i] = B[2] * MP.B[K + i]    # spin-down
        end

        copyto!(FC.B[i], F[1])
        copyto!(FC.B[K + i], F[2])

        rmul!(F[1], MP.B[i], ws)
        rmul!(F[2], MP.B[K + i], ws)
    end

    return F, MP, FC
end

function run_full_propagation_reverse(
    MP::Cluster{C}, ws::LDRWorkspace{T,E};
    V = size(MP.B[1]),
    F = ldrs(Matrix(1.0I, V), 2),
    FC = Cluster(B = ldrs(Matrix(1.0I, V), 2 * K))
) where {C, T, E}
    """
        Propagate the full space-time lattice in the reverse order
    """
    K = div(length(MP.B), 2)

    for i in K : -1 : 1

        copyto!(FC.B[i], F[1])
        copyto!(FC.B[K + i], F[2])

        rmul!(F[1], MP.B[i], ws)
        rmul!(F[2], MP.B[K + i], ws)
    end

    return F, FC
end
