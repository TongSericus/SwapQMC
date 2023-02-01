include("./swap_IonHubbard.jl")

const worker_id = parse(Int64, get(ENV, "SLURM_ARRAY_TASK_ID", 1))

const id = mod(worker_id - 1, 10) + 1
const file_id = div(worker_id - 1, 10) + 1
@show file_id

const U = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.2, 2.5, 3.0]
const μ = [0.10080122704425552, 0.2504849954395389, 0.39958101548246694, 0.49758263082388693, 0.59768144172065, 0.7468072650698482, 0.9978202530373774, 1.0990170934998647, 1.2452214895048563, 1.5123961647637385]

const system = IonicHubbard(
    # (Nx, Ny), (N_up, N_dn)
    (6, 6), (18, 18),
    # t, U, Δ
    1.0, U[id], 0.5,
    # μ
    μ[id],
    # β, L
    12.0, 240
)

const qmc = QMC(
    system,
    # number of warm-ups, samples and measurement interval
    50, Int64(1e3), 20,
    # stablization and update interval
    10, 10
)

# assign subsystem indices
const Aidx = collect(1:18)
const extsys = ExtendedSystem(system, Aidx)

seed = file_id + 1234
@show seed
Random.seed!(seed)

const path = "GCE_IonHub_Lx$(system.Ns[1])Ly$(system.Ns[2])_LA$(extsys.LA)"
const filename = "denom_U$(system.U)_Delta$(system.Δ)_beta$(system.β)_$(file_id).jld"

run_swap_gce(extsys, qmc, 1, path, filename)