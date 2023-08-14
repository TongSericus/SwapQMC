include("./qmc_incremental_SSH_gs.jl")

const δt_list = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
const Nₖ = 5
const λₖ_list = collect(0.0:0.2:0.8)

## extract array ID from the environment
const worker_id = parse(Int64, get(ENV, "SLURM_ARRAY_TASK_ID", 1))
## assign ID to files
const id = mod(worker_id - 1, length(δt_list)) + 1
const job_id = div(worker_id - 1, length(δt_list)) + 1
const λid = mod(job_id - 1, Nₖ) + 1
const file_id = div(job_id - 1, Nₖ) + 1

const L = 42
const T = hopping_matrix_ssh_1d(L, 1.0, δt_list[id], isOBC=false)

const system = GenericHubbard(
    # (Nx, Ny), (N_up, N_dn)
    (L, 1, 1), (21, 21),
    # t, U
    T, 4.0,
    # μ
    0.0,
    # β, L
    32.0, 320,
    # data type of the system
    sys_type=ComplexF64,
    # if use charge decomposition
    useChargeHST=true,
    # if use first-order Trotteriaztion
    useFirstOrderTrotter=false
)

const qmc = QMC(
    system,
    # number of warm-ups, samples and measurement interval
    512, 2048, 5,
    # stablization and update interval
    10, 10,
    # if force spin symmetry
    forceSymmetry=false,
    # debugging flag
    saveRatio=false
)

seed = 1234 + file_id
@show seed
Random.seed!(seed)

const φ₀_up = trial_wf_free(system, 1, T)
const φ₀ = [φ₀_up, copy(φ₀_up)]

const Aidx = δt_list[id] <= 0 ? collect(2:div(L,2)+1) : collect(1:div(L,2))
const extsys = ExtendedSystem(system, Aidx, subsysOrdering=false)

const λₖ = λₖ_list[λid]

path = "./data/"
filename = "GS_PBC_L$(system.Ns[1])_dt$(δt_list[id])_U$(system.U)_lambda$(λₖ)_beta$(system.β)_seed$(seed).jld"
@time run_incremental_sampling_gs(extsys, qmc, φ₀, λₖ, Nₖ, path, filename)