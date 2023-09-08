using SwapQMC, Test

#######################
##### Test Module #####
#######################

# finite temperature test
@testset "SwapQMC_FT" begin
    ##### Regular 2D Hubbard Model #####

    ### standard sweep ###
    Lx, Ly = 4, 4
    T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)

    system = GenericHubbard(
        # (Nx, Ny), (N_up, N_dn)
        (Lx, Ly, 1), (8, 8),
        # t, U
        T, 4.0,
        # μ
        0.0,
        # β, L
        16.0, 160,
        # if use charge decomposition
        useChargeHST = true,
        # if use first-order Trotteriaztion
        useFirstOrderTrotter = false
    )

    qmc = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        100, 200, 10,
        # stablization and update interval
        8, 8,
        # debugging flag
        saveRatio=true
    )

    # create a random walker and sweep
    walker = HubbardGCWalker(system, qmc)
    auxfield = copy(walker.auxfield)
    sweep!(system, qmc, walker)

    # create a replica
    walker′ = HubbardGCWalker(system, qmc, auxfield=walker.auxfield)
    # and test if the Green's function is correct at the end of sweep
    @test walker.G[1] ≈ walker′.G[1]

    # pick a random point in the space-time lattice
    ind = rand(1 : system.V * system.L)
    # and copy
    @. auxfield[1:ind-1] = walker.auxfield[1:ind-1]
    walker′ = HubbardGCWalker(system, qmc, auxfield=auxfield)
    auxfield[ind] *= -1
    walker″ = HubbardGCWalker(system, qmc, auxfield=auxfield)
    r = exp(sum(walker″.weight - walker′.weight))
    # and test if the correct Metropolis ratio is produced
    @test r ≈ walker.tmp_r[ind]

    ### replica sweep ###
    walker1 = HubbardGCWalker(system, qmc)
    auxfield = copy(walker1.auxfield)
    walker2 = HubbardGCWalker(system, qmc)
    Aidx = collect(1 : div(system.V,2))
    extsys = ExtendedSystem(system, Aidx, subsysOrdering=true)
    replica = Replica(extsys, walker1, walker2)

    sweep!(system, qmc, replica, walker1, 1)
    walker′ = HubbardGCWalker(system, qmc, auxfield=walker1.auxfield)
    replica′ = Replica(extsys, walker′, walker2)

    @test replica.logdetGA[] ≈ replica′.logdetGA[]
    @test replica.GA⁻¹ ≈ replica′.GA⁻¹

    # pick a random point in the space-time lattice
    ind = rand(1 : system.V * system.L)
    # and copy
    @. auxfield[1:ind-1] = walker1.auxfield[1:ind-1]
    walker′ = HubbardGCWalker(system, qmc, auxfield=auxfield)
    auxfield[ind] *= -1
    walker″ = HubbardGCWalker(system, qmc, auxfield=auxfield)
    replica′ = Replica(extsys, walker′, walker2)
    replica″ = Replica(extsys, walker″, walker2)
    logr = sum(walker″.weight - walker′.weight) - 2*(replica″.logdetGA[] - replica′.logdetGA[])
    @test exp(logr) ≈ walker1.tmp_r[ind]

    # proceed to the next walker
    auxfield = copy(walker2.auxfield)
    sweep!(system, qmc, replica, walker2, 2)
    walker′ = HubbardGCWalker(system, qmc, auxfield=walker2.auxfield)
    replica′ = Replica(extsys, walker1, walker′)

    @test replica.logdetGA[] ≈ replica′.logdetGA[]
    @test replica.GA⁻¹ ≈ replica′.GA⁻¹

    # same, pick a random point in the space-time lattice
    ind = rand(1 : system.V * system.L)
    # and copy
    @. auxfield[1:ind-1] = walker2.auxfield[1:ind-1]
    walker′ = HubbardGCWalker(system, qmc, auxfield=auxfield)
    auxfield[ind] *= -1
    walker″ = HubbardGCWalker(system, qmc, auxfield=auxfield)
    replica′ = Replica(extsys, walker1, walker′)
    replica″ = Replica(extsys, walker1, walker″)
    logr = sum(walker″.weight - walker′.weight) - 2*(replica″.logdetGA[] - replica′.logdetGA[])
    @test exp(logr) ≈ walker2.tmp_r[ind]

    ##### 1D SSH-Hubbard model with modulated hopping #####

    ### standard sweep ###
    L = 12
    T = hopping_matrix_ssh_1d(L, 1.0, -0.1)
    system = GenericHubbard(
        # (Nx, Ny), (N_up, N_dn)
        (L, 1, 1), (6, 6),
        # t, U
        T, 4.0,
        # μ
        0.0,
        # β, L
        16.0, 160,
        # if use charge decomposition
        useChargeHST = true,
        # if use first-order Trotteriaztion
        useFirstOrderTrotter = false
    )
    qmc = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        100, 200, 10,
        # stablization and update interval
        8, 8,
        # debugging flag
        saveRatio=true
    )

    # create a random walker and sweep
    walker = HubbardGCWalker(system, qmc)
    auxfield = copy(walker.auxfield)
    sweep!(system, qmc, walker)

    # create a replica
    walker′ = HubbardGCWalker(system, qmc, auxfield=walker.auxfield)
    # and test if the Green's function is correct at the end of sweep
    @test walker.G[1] ≈ walker′.G[1]

    # pick a random point in the space-time lattice
    ind = rand(1 : system.V * system.L)
    # and copy
    @. auxfield[1:ind-1] = walker.auxfield[1:ind-1]
    walker′ = HubbardGCWalker(system, qmc, auxfield=auxfield)
    auxfield[ind] *= -1
    walker″ = HubbardGCWalker(system, qmc, auxfield=auxfield)
    r = exp(sum(walker″.weight - walker′.weight))
    # and test if the correct Metropolis ratio is produced
    @test r ≈ walker.tmp_r[ind]

    ### replica sweep ###
    walker1 = HubbardGCWalker(system, qmc)
    auxfield = copy(walker1.auxfield)
    walker2 = HubbardGCWalker(system, qmc)
    Aidx = collect(1 : div(system.V,2))
    extsys = ExtendedSystem(system, Aidx, subsysOrdering=true)
    replica = Replica(extsys, walker1, walker2)

    sweep!(system, qmc, replica, walker1, 1)
    walker′ = HubbardGCWalker(system, qmc, auxfield=walker1.auxfield)
    replica′ = Replica(extsys, walker′, walker2)

    @test replica.logdetGA[] ≈ replica′.logdetGA[]
    @test replica.GA⁻¹ ≈ replica′.GA⁻¹

    # pick a random point in the space-time lattice
    ind = rand(1 : system.V * system.L)
    # and copy
    @. auxfield[1:ind-1] = walker1.auxfield[1:ind-1]
    walker′ = HubbardGCWalker(system, qmc, auxfield=auxfield)
    auxfield[ind] *= -1
    walker″ = HubbardGCWalker(system, qmc, auxfield=auxfield)
    replica′ = Replica(extsys, walker′, walker2)
    replica″ = Replica(extsys, walker″, walker2)
    logr = sum(walker″.weight - walker′.weight) - 2*(replica″.logdetGA[] - replica′.logdetGA[])
    @test exp(logr) ≈ walker1.tmp_r[ind]

    # proceed to the next walker
    auxfield = copy(walker2.auxfield)
    sweep!(system, qmc, replica, walker2, 2)
    walker′ = HubbardGCWalker(system, qmc, auxfield=walker2.auxfield)
    replica′ = Replica(extsys, walker1, walker′)

    @test replica.logdetGA[] ≈ replica′.logdetGA[]
    @test replica.GA⁻¹ ≈ replica′.GA⁻¹

    # same, pick a random point in the space-time lattice
    ind = rand(1 : system.V * system.L)
    # and copy
    @. auxfield[1:ind-1] = walker2.auxfield[1:ind-1]
    walker′ = HubbardGCWalker(system, qmc, auxfield=auxfield)
    auxfield[ind] *= -1
    walker″ = HubbardGCWalker(system, qmc, auxfield=auxfield)
    replica′ = Replica(extsys, walker1, walker′)
    replica″ = Replica(extsys, walker1, walker″)
    logr = sum(walker″.weight - walker′.weight) - 2*(replica″.logdetGA[] - replica′.logdetGA[])
    @test exp(logr) ≈ walker2.tmp_r[ind]
    
end

### functions for ground state test ###

# use BigFloat to compute the proagator precisely
BigMatrix(A::LDR) = big.(A.L) * Diagonal(big.(A.d)) * big.(A.R)

function compute_gs_projected_matrix(walker::HubbardWalker)
    B_bra = big.(walker.φ₀T[1]) * BigMatrix(walker.Fl[1])
    B_ket = BigMatrix(walker.Fr[1]) * big.(walker.φ₀[1])
    return log(abs(det(B_bra * B_ket)))
end

function compute_ratio_regular(
    system::Hubbard, qmc::QMC,
    auxfield::AbstractArray{Int}, idx::Tuple{Int, Int},
    φ₀::AbstractVector{Wf}
) where Wf
    idx_x, idx_t = idx

    walker′ = HubbardWalker(system, qmc, φ₀, auxfield=auxfield)
    auxfield[idx_x, idx_t] *= -1
    walker″ = HubbardWalker(system, qmc, φ₀, auxfield=auxfield)

    logdetB = compute_gs_projected_matrix(walker′)
    logdetB′ = compute_gs_projected_matrix(walker″)
    return exp(2*(logdetB′ - logdetB))
end

function compute_ratio_replica(
    extsys::ExtendedSystem, qmc::QMC,
    auxfield::AbstractArray{Int}, idx::Tuple{Int, Int}, 
    walker::HubbardWalker; ridx::Int = 1, λₖ::Float64 = 1.0
)
    system = extsys.system
    idx_x, idx_t = idx
    φ₀ = walker.φ₀

    walker′ = HubbardWalker(system, qmc, φ₀, auxfield=auxfield)
    auxfield[idx_x, idx_t] *= -1
    walker″ = HubbardWalker(system, qmc, φ₀, auxfield=auxfield)
    ridx == 1 ? (
            replica′ = Replica(extsys, walker′, walker);
            replica″ = Replica(extsys, walker″, walker)
        ) :
        (
            replica′ = Replica(extsys, walker, walker′);
            replica″ = Replica(extsys, walker, walker″)
        )

    logdetB = compute_gs_projected_matrix(walker′)
    logdetB′ = compute_gs_projected_matrix(walker″)
    logr = 2*(logdetB′ - logdetB) + 2*replica′.logdetGA[] - 2*replica″.logdetGA[]

    return exp(2*(logdetB′ - logdetB)) * exp(2*(replica′.logdetGA[] - replica″.logdetGA[]))^λₖ
end

function test_regular_sweep(system::Hubbard, qmc::QMC, walker::HubbardWalker)

    r = zeros(Float64, 4)
    idx = zeros(Int, 4)

    Θ = div(qmc.K,2)
    θ = div(system.L,2)

    ## Test θ -> 2θ ##
    auxfield = copy(walker.auxfield)
    sweep!_symmetric(system, qmc, walker, collect(Θ+1:2Θ))
    # pick a random point in time [θ:2θ]
    idx_t = rand(θ+1:2θ)
    idx_x = rand(1:system.V)
    idx[1] = (idx_t-θ-1)*system.V + idx_x
    @. auxfield[:, θ+1:idx_t-1] = walker.auxfield[:, θ+1:idx_t-1]
    @. auxfield[1:idx_x-1, idx_t] = walker.auxfield[1:idx_x-1, idx_t]
    r[1] = compute_ratio_regular(system, qmc, auxfield, (idx_x, idx_t), walker.φ₀)

    ## Test 2θ -> Θ ##
    auxfield = copy(walker.auxfield)
    sweep!_symmetric(system, qmc, walker, collect(2Θ:-1:Θ+1))
    # pick a random point in time [θ:2θ]
    idx_t = rand(θ+1:2θ)
    idx_x = rand(1:system.V)
    idx[2] = (2θ-idx_t)*system.V + idx_x + θ*system.V
    @. auxfield[:, idx_t+1:2θ] = walker.auxfield[:, idx_t+1:2θ]
    @. auxfield[1:idx_x-1, idx_t] = walker.auxfield[1:idx_x-1, idx_t]
    r[2] = compute_ratio_regular(system, qmc, auxfield, (idx_x, idx_t), walker.φ₀)

    ## Test Θ -> 0 ##
    auxfield = copy(walker.auxfield)
    sweep!_symmetric(system, qmc, walker, collect(Θ:-1:1))
    # pick a random point in time [1:Θ]
    idx_t = rand(1:θ)
    idx_x = rand(1:system.V)
    idx[3] = (θ-idx_t)*system.V + idx_x + 2*θ*system.V
    @. auxfield[:, θ:-1:idx_t+1] = walker.auxfield[:, θ:-1:idx_t+1]
    @. auxfield[1:idx_x-1, idx_t] = walker.auxfield[1:idx_x-1, idx_t]
    r[3] = compute_ratio_regular(system, qmc, auxfield, (idx_x, idx_t), walker.φ₀)

    ## Test 0 -> Θ ##
    auxfield = copy(walker.auxfield)
    sweep!_symmetric(system, qmc, walker, collect(1:Θ))
    # pick a random point in time [1:Θ]
    idx_t = rand(1:θ)
    idx_x = rand(1:system.V)
    idx[4] = (idx_t-1)*system.V + idx_x + 3*θ*system.V
    @. auxfield[:, 1:idx_t-1] = walker.auxfield[:, 1:idx_t-1]
    @. auxfield[1:idx_x-1, idx_t] = walker.auxfield[1:idx_x-1, idx_t]
    r[4] = compute_ratio_regular(system, qmc, auxfield, (idx_x, idx_t), walker.φ₀)

    return r, idx
end

function test_replica_sweep(extsys::ExtendedSystem, qmc::QMC, replica::Replica; λₖ::Float64 = 1.0)
    
    r = zeros(Float64, 4)
    idx = zeros(Int, 4)

    system = extsys.system
    Θ = div(qmc.K,2)
    θ = div(system.L,2)

    walker1 = replica.walker1
    walker2 = replica.walker2

    ### test the first replica in the forward direction ###
    auxfield = copy(walker1.auxfield)
    sweep!_symmetric(system, qmc, replica, walker1, 1, collect(Θ+1:2Θ))

    # pick a random point in time [θ:2θ]
    idx_t = rand(θ+1:2θ)
    idx_x = rand(1:system.V)
    idx[1] = (idx_t-θ-1)*system.V + idx_x
    @. auxfield[:, θ+1:idx_t-1] = walker1.auxfield[:, θ+1:idx_t-1]
    @. auxfield[1:idx_x-1, idx_t] = walker1.auxfield[1:idx_x-1, idx_t]
    r[1] = compute_ratio_replica(extsys, qmc, auxfield, (idx_x, idx_t), walker2, ridx=1, λₖ=λₖ)

    ### test the first replica in the backward direction ###
    auxfield = copy(walker1.auxfield)
    sweep!_symmetric(system, qmc, replica, walker1, 1, collect(Θ:-1:1))
    # pick a random point in time [θ:2θ]
    idx_t = rand(1:θ)
    idx_x = rand(1:system.V)
    idx[2] = (θ - idx_t)*system.V + idx_x + div(system.V*system.L,2)
    @. auxfield[:, idx_t+1:θ] = walker1.auxfield[:, idx_t+1:θ]
    @. auxfield[1:idx_x-1, idx_t] = walker1.auxfield[1:idx_x-1, idx_t]
    r[2] = compute_ratio_replica(extsys, qmc, auxfield, (idx_x, idx_t), walker2, ridx=1, λₖ=λₖ)

    ### switch to the second replica and perform the same test ###
    jump_replica!(replica, 1)

    ## forward direction ##
    auxfield = copy(walker2.auxfield)
    sweep!_symmetric(system, qmc, replica, walker2, 2, collect(Θ+1:2Θ))

    # pick a random point in time [θ:2θ]
    idx_t = rand(θ+1:2θ)
    idx_x = rand(1:system.V)
    idx[3] = (idx_t-θ-1)*system.V + idx_x
    @. auxfield[:, θ+1:idx_t-1] = walker2.auxfield[:, θ+1:idx_t-1]
    @. auxfield[1:idx_x-1, idx_t] = walker2.auxfield[1:idx_x-1, idx_t]
    r[3] = compute_ratio_replica(extsys, qmc, auxfield, (idx_x, idx_t), walker1, ridx=2, λₖ=λₖ)

    ## backward direction ##
    auxfield = copy(walker2.auxfield)
    sweep!_symmetric(system, qmc, replica, walker2, 2, collect(Θ:-1:1))
    # pick a random point in time [θ:2θ]
    idx_t = rand(1:θ)
    idx_x = rand(1:system.V)
    idx[4] = (θ - idx_t)*system.V + idx_x + div(system.V*system.L,2)
    @. auxfield[:, idx_t+1:θ] = walker2.auxfield[:, idx_t+1:θ]
    @. auxfield[1:idx_x-1, idx_t] = walker2.auxfield[1:idx_x-1, idx_t]
    r[4] = compute_ratio_replica(extsys, qmc, auxfield, (idx_x, idx_t), walker1, ridx=2, λₖ=λₖ)

    return r, idx
end

# ground state test
@testset "SwapQMC_GS" begin
    ##### Attractive 2D Hubbard Model with Charge Decomposition #####
    Lx, Ly = 4, 4
    T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)

    system = GenericHubbard(
        # (Nx, Ny), (N_up, N_dn)
        (Lx, Ly, 1), (7, 7),
        # t, U
        T, -4.0,
        # μ
        0.0,
        # β, L
        16.0, 160,
        # data type of the system
        sys_type=Float64,
        # if use charge decomposition
        useChargeHST=true,
        # if use first-order Trotteriaztion
        useFirstOrderTrotter=false
    )

    qmc = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        500, 2000, 10,
        # stablization and update interval
        5, 5,
        # debugging flag
        saveRatio=true
    )

    qmc_nosave = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        500, 2000, 10,
        # stablization and update interval
        5, 5,
        # debugging flag
        saveRatio=false
    )

    φ₀_up = trial_wf_free(system, 1, T)
    φ₀ = [φ₀_up, copy(φ₀_up)]

    ### test the regular sweep ###
    walker = HubbardWalker(system, qmc, φ₀)
    # thermalize the walker without saving ratios
    sweep!(system, qmc_nosave, walker, loop_number=10)

    r, idx = test_regular_sweep(system, qmc, walker)
    @test r[1] ≈ walker.tmp_r[idx[1]]
    @test r[2] ≈ walker.tmp_r[idx[2]]
    @test r[3] ≈ walker.tmp_r[idx[3]]
    @test r[4] ≈ walker.tmp_r[idx[4]]

    ### test the replica sweep ###
    extsys = ExtendedSystem(system, collect(1:8), subsysOrdering=false)

    walker1 = HubbardWalker(system, qmc, φ₀)
    walker2 = HubbardWalker(system, qmc, φ₀)
    replica = Replica(extsys, walker1, walker2)
    sweep!(system, qmc_nosave, replica, walker1, 1, loop_number=10, jumpReplica=true)
    sweep!(system, qmc_nosave, replica, walker2, 2, loop_number=10, jumpReplica=true)

    r, idx = test_replica_sweep(extsys, qmc, replica)
    @test r[1] ≈ walker1.tmp_r[idx[1]]
    @test r[2] ≈ walker1.tmp_r[idx[2]]
    @test r[3] ≈ walker2.tmp_r[idx[3]]
    @test r[4] ≈ walker2.tmp_r[idx[4]]

    ##### Attractive 2D Hubbard Model with Spin Decomposition #####
    Lx, Ly = 4, 4
    T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)

    system = GenericHubbard(
        # (Nx, Ny), (N_up, N_dn)
        (Lx, Ly, 1), (7, 7),
        # t, U
        T, -4.0,
        # μ
        0.0,
        # β, L
        16.0, 160,
        # data type of the system
        sys_type=ComplexF64,
        # if use charge decomposition
        useChargeHST=false,
        # if use first-order Trotteriaztion
        useFirstOrderTrotter=false
    )

    qmc = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        500, 2000, 10,
        # stablization and update interval
        5, 5,
        # if force the spin symmetry
        forceSymmetry=true,
        # debugging flag
        saveRatio=true
    )

    qmc_nosave = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        500, 2000, 10,
        # stablization and update interval
        5, 5,
        # if force the spin symmetry
        forceSymmetry=true,
        # debugging flag
        saveRatio=false
    )

    φ₀_up = trial_wf_free(system, 1, T)
    φ₀ = [φ₀_up, copy(φ₀_up)]

    ### test the regular sweep ###
    walker = HubbardWalker(system, qmc, φ₀)
    # thermalize the walker without saving ratios
    sweep!(system, qmc_nosave, walker, loop_number=10)

    r, idx = test_regular_sweep(system, qmc, walker)
    @test r[1] ≈ walker.tmp_r[idx[1]]
    @test r[2] ≈ walker.tmp_r[idx[2]]
    @test r[3] ≈ walker.tmp_r[idx[3]]
    @test r[4] ≈ walker.tmp_r[idx[4]]

    ### test the replica sweep ###
    extsys = ExtendedSystem(system, collect(1:8), subsysOrdering=false)

    walker1 = HubbardWalker(system, qmc, φ₀)
    walker2 = HubbardWalker(system, qmc, φ₀)
    replica = Replica(extsys, walker1, walker2)
    sweep!(system, qmc_nosave, replica, walker1, 1, loop_number=10, jumpReplica=true)
    sweep!(system, qmc_nosave, replica, walker2, 2, loop_number=10, jumpReplica=true)

    r, idx = test_replica_sweep(extsys, qmc, replica)
    @test r[1] ≈ walker1.tmp_r[idx[1]]
    @test r[2] ≈ walker1.tmp_r[idx[2]]
    @test r[3] ≈ walker2.tmp_r[idx[3]]
    @test r[4] ≈ walker2.tmp_r[idx[4]]

    ##### Repulsive 2D Hubbard Model with Charge Decomposition #####
    Lx, Ly = 4, 4
    T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)

    system = GenericHubbard(
        # (Nx, Ny), (N_up, N_dn)
        (Lx, Ly, 1), (8, 8),
        # t, U
        T, 4.0,
        # μ
        0.0,
        # β, L
        16.0, 160,
        # data type of the system
        sys_type=ComplexF64,
        # if use charge decomposition
        useChargeHST=true,
        # if use first-order Trotteriaztion
        useFirstOrderTrotter=false
    )

    qmc = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        500, 2000, 10,
        # stablization and update interval
        5, 5,
        # if force the spin symmetry
        forceSymmetry=true,
        # debugging flag
        saveRatio=true
    )

    qmc_nosave = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        500, 2000, 10,
        # stablization and update interval
        5, 5,
        # if force the spin symmetry
        forceSymmetry=true,
        # debugging flag
        saveRatio=false
    )

    φ₀_up = trial_wf_free(system, 1, T)
    φ₀ = [φ₀_up, copy(φ₀_up)]

    ### test the regular sweep ###
    walker = HubbardWalker(system, qmc, φ₀)
    # thermalize the walker without saving ratios
    sweep!(system, qmc_nosave, walker, loop_number=10)

    r, idx = test_regular_sweep(system, qmc, walker)
    @test r[1] ≈ walker.tmp_r[idx[1]]
    @test r[2] ≈ walker.tmp_r[idx[2]]
    @test r[3] ≈ walker.tmp_r[idx[3]]
    @test r[4] ≈ walker.tmp_r[idx[4]]

    ### test the replica sweep ###
    extsys = ExtendedSystem(system, collect(1:8), subsysOrdering=false)

    walker1 = HubbardWalker(system, qmc, φ₀)
    walker2 = HubbardWalker(system, qmc, φ₀)
    replica = Replica(extsys, walker1, walker2)
    sweep!(system, qmc_nosave, replica, walker1, 1, loop_number=10, jumpReplica=true)
    sweep!(system, qmc_nosave, replica, walker2, 2, loop_number=10, jumpReplica=true)

    r, idx = test_replica_sweep(extsys, qmc, replica)
    @test r[1] ≈ walker1.tmp_r[idx[1]]
    @test r[2] ≈ walker1.tmp_r[idx[2]]
    @test r[3] ≈ walker2.tmp_r[idx[3]]
    @test r[4] ≈ walker2.tmp_r[idx[4]]

    ### test the incremental replica sweep ###
    walker1 = HubbardWalker(system, qmc, φ₀)
    walker2 = HubbardWalker(system, qmc, φ₀)
    # a random thermaldynamic integration variable
    λₖ = rand()
    replica = Replica(extsys, walker1, walker2, λₖ=λₖ)
    sweep!(system, qmc_nosave, replica, walker1, 1, loop_number=10, jumpReplica=true)
    sweep!(system, qmc_nosave, replica, walker2, 2, loop_number=10, jumpReplica=true)

    r, idx = test_replica_sweep(extsys, qmc, replica, λₖ=λₖ)
    @test r[1] ≈ walker1.tmp_r[idx[1]]
    @test r[2] ≈ walker1.tmp_r[idx[2]]
    @test r[3] ≈ walker2.tmp_r[idx[3]]
    @test r[4] ≈ walker2.tmp_r[idx[4]]
end

# ground state test with Hartree Fock trial wavefunction
@testset "SwapQMC_GS_HFwavefunc" begin
    ##### Attractive 2D Hubbard Model with Charge Decomposition #####
    Lx, Ly = 4, 4
    T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)

    system = GenericHubbard(
        # (Nx, Ny), (N_up, N_dn)
        (Lx, Ly, 1), (7, 7),
        # t, U
        T, -4.0,
        # μ
        0.0,
        # β, L
        16.0, 160,
        # data type of the system
        sys_type=Float64,
        # if use charge decomposition
        useChargeHST=true,
        # if use first-order Trotteriaztion
        useFirstOrderTrotter=false
    )

    qmc = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        500, 2000, 10,
        # stablization and update interval
        5, 5,
        # debugging flag
        saveRatio=true
    )

    qmc_nosave = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        500, 2000, 10,
        # stablization and update interval
        5, 5,
        # debugging flag
        saveRatio=false
    )

    φ₀ = trial_wf_HF(system, ϵ=1e-10)

    ### test the regular sweep ###
    walker = HubbardWalker(system, qmc, φ₀)
    # thermalize the walker without saving ratios
    sweep!(system, qmc_nosave, walker, loop_number=10)

    r, idx = test_regular_sweep(system, qmc, walker)
    @test r[1] ≈ walker.tmp_r[idx[1]]
    @test r[2] ≈ walker.tmp_r[idx[2]]
    @test r[3] ≈ walker.tmp_r[idx[3]]
    @test r[4] ≈ walker.tmp_r[idx[4]]

    ### test the replica sweep ###
    extsys = ExtendedSystem(system, collect(1:8), subsysOrdering=false)

    walker1 = HubbardWalker(system, qmc, φ₀)
    walker2 = HubbardWalker(system, qmc, φ₀)
    replica = Replica(extsys, walker1, walker2)
    sweep!(system, qmc_nosave, replica, walker1, 1, loop_number=10, jumpReplica=true)
    sweep!(system, qmc_nosave, replica, walker2, 2, loop_number=10, jumpReplica=true)

    r, idx = test_replica_sweep(extsys, qmc, replica)
    @test r[1] ≈ walker1.tmp_r[idx[1]]
    @test r[2] ≈ walker1.tmp_r[idx[2]]
    @test r[3] ≈ walker2.tmp_r[idx[3]]
    @test r[4] ≈ walker2.tmp_r[idx[4]]

    ##### Attractive 2D Hubbard Model with Spin Decomposition #####
    Lx, Ly = 4, 4
    T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)

    system = GenericHubbard(
        # (Nx, Ny), (N_up, N_dn)
        (Lx, Ly, 1), (7, 7),
        # t, U
        T, -4.0,
        # μ
        0.0,
        # β, L
        16.0, 160,
        # data type of the system
        sys_type=ComplexF64,
        # if use charge decomposition
        useChargeHST=false,
        # if use first-order Trotteriaztion
        useFirstOrderTrotter=false
    )

    qmc = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        500, 2000, 10,
        # stablization and update interval
        5, 5,
        # if force the spin symmetry
        forceSymmetry=true,
        # debugging flag
        saveRatio=true
    )

    qmc_nosave = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        500, 2000, 10,
        # stablization and update interval
        5, 5,
        # if force the spin symmetry
        forceSymmetry=true,
        # debugging flag
        saveRatio=false
    )

    φ₀ = trial_wf_HF(system, ϵ=1e-10)

    ### test the regular sweep ###
    walker = HubbardWalker(system, qmc, φ₀)
    # thermalize the walker without saving ratios
    sweep!(system, qmc_nosave, walker, loop_number=10)

    r, idx = test_regular_sweep(system, qmc, walker)
    @test r[1] ≈ walker.tmp_r[idx[1]]
    @test r[2] ≈ walker.tmp_r[idx[2]]
    @test r[3] ≈ walker.tmp_r[idx[3]]
    @test r[4] ≈ walker.tmp_r[idx[4]]

    ### test the replica sweep ###
    extsys = ExtendedSystem(system, collect(1:8), subsysOrdering=false)

    walker1 = HubbardWalker(system, qmc, φ₀)
    walker2 = HubbardWalker(system, qmc, φ₀)
    replica = Replica(extsys, walker1, walker2)
    sweep!(system, qmc_nosave, replica, walker1, 1, loop_number=10, jumpReplica=true)
    sweep!(system, qmc_nosave, replica, walker2, 2, loop_number=10, jumpReplica=true)

    r, idx = test_replica_sweep(extsys, qmc, replica)
    @test r[1] ≈ walker1.tmp_r[idx[1]]
    @test r[2] ≈ walker1.tmp_r[idx[2]]
    @test r[3] ≈ walker2.tmp_r[idx[3]]
    @test r[4] ≈ walker2.tmp_r[idx[4]]
end
