include("./swap.jl")

# define the kinetic matrix
function hopping_matrix_sshHubbard_1D(
    L::Int, t::Float64, δt::Float64; 
    isOBC::Bool = true
)
    T = zeros(L, L)

    isOBC ? begin
        hop_ind_left = [CartesianIndex(i+1, (i+1)%L+1) for i in 0 : L-2]
        hop_ind_right = [CartesianIndex((i+1)%L+1, i+1) for i in 0 : L-2]
        hop_amp = [-t-δt*(-1)^i for i in 0 : L-2]
    end :
    # periodic boundary condition
    begin
        hop_ind_left = [CartesianIndex(i+1, (i+1)%L+1) for i in 0 : L-1]
        hop_ind_right = [CartesianIndex((i+1)%L+1, i+1) for i in 0 : L-1]
        hop_amp = [-t-δt*(-1)^i for i in 0 : L-1]
    end

    @views T[hop_ind_left] = hop_amp
    @views T[hop_ind_right] = hop_amp
    
    return T
end

const worker_id = parse(Int64, get(ENV, "SLURM_ARRAY_TASK_ID", 1))

const id = mod(worker_id - 1, 5) + 1
const file_id = div(worker_id - 1, 5) + 1
@show file_id

δt_list = [-0.5, -0.4, -0.3, -0.2, -0.1]

const L = 12
const nup, ndn = div(L, 2), div(L, 2)
const t, δt = 1.0, -0.1
const T = hopping_matrix_sshHubbard_1D(L, t, δt, isOBC=true)

const system = GenericHubbard(
    # (Nx, Ny, Nz), (N_up, N_dn)
    (L, 1, 1), (nup, ndn),
    # t, U
    T, 4.0,
    # μ
    0.0,
    # β, L
    16.0, 160,
    # if use charge decomposition
    useComplexHST=true,
    # if use first-order Trotteriaztion
    useFirstOrderTrotter=false
)

const qmc = QMC(
    system,
    # number of warm-ups, samples and measurement interval
    100, 2000, 10,
    # stablization and update interval
    8, 8
)

# assign subsystem indices
const Aidx = δt > 0 ? collect(1 : div(L,2)) : collect(1 : div(L,2)+1)
const extsys = ExtendedSystem(system, Aidx, subsysOrdering=true)

seed = file_id + 1234
@show seed
Random.seed!(seed)

const path = "../data/SSH_L$(system.Ns[1])_OBC"
const filename = "denom_U$(system.U)_dt$(δt)_beta$(system.β)_$(seed).jld"

@time run_swap_gce(extsys, qmc, 1, path, filename)
