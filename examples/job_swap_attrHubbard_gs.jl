const Lx, Ly = 8, 8
const T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)
const U_list = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -10.0]

# extract array ID from the environment
const worker_id = parse(Int64, get(ENV, "SLURM_ARRAY_TASK_ID", 1))
# assign ID to files
const id = mod(worker_id - 1, 6) + 1
const file_id = div(worker_id - 1, 6) + 1
@show file_id

const system = GenericHubbard(
    # (Nx, Ny), (N_up, N_dn)
    (Lx, Ly, 1), (32, 32),
    # t, U
    T, U_list[id],
    # μ
    0.5,
    # β, L
    18.0, 180,
    # data type of the system
    sys_type=ComplexF64,
    # if use charge decomposition
    useChargeHST=false,
    # if use first-order Trotteriaztion
    useFirstOrderTrotter=false
)

const qmc = QMC(
    system,
    # number of warm-ups, samples and measurement interval
    512, 2048, 6,
    # stablization and update interval
    10, 10,
    # if force spin symmetry
    forceSymmetry=true,
    # debugging flag
    saveRatio=false
)

seed = 2434 + file_id
@show seed
Random.seed!(seed)

const φ₀_up = trial_wf_free(system, 1, T)
const φ₀ = [φ₀_up, copy(φ₀_up)]

const Aidx = collect(1:32)
const extsys = ExtendedSystem(system, Aidx, subsysOrdering=false)

path = "../data/AttrHubbard_Lx8Ly8/LA$(extsys.LA)"
filename = "GS_denom_U$(system.U)_N$(sum(system.N))_Lx$(system.Ns[1])_Ly$(system.Ns[2])_LA$(extsys.LA)_beta$(system.β)_seed$(seed).jld"
@time run_swap_sampling_gs(extsys, qmc, φ₀, 2, path, filename)
