"""
   Operations for extended systems
"""

function run_full_propagation(
    auxfield::AbstractMatrix{Int64}, system::ExtendedSystem, qmc::QMC, ws::LDRWorkspace{T, E}; 
    K = qmc.K, stab_interval = qmc.stab_interval, K_interval = qmc.K_interval
) where {T, E}
    V = system.V

    B = [Matrix{Float64}(1.0I, V, V), Matrix{Float64}(1.0I, V, V)]
    Bext = [Matrix{Float64}(1.0I, Vext, Vext), Matrix{Float64}(1.0I, Vext, Vext)]
    MP = Cluster(V, 2 * K)

    F = ldrs(B[1], 2)
    FC = Cluster(B = ldrs(B[1], 2 * K))

    for i in 1 : K

        for j = 1 : K_interval[i]
            @views σ = auxfield[:, (i - 1) * stab_interval + j]
            singlestep_matrix!(B[1], B[2], Bext[1], Bext[2], 0, σ, system, tmpmat = ws.M)
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