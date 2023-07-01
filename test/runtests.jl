using SwapQMC, Test

###############################################
##### Define Kinetic Matrices for Testing #####
###############################################

function hopping_matrix_Hubbard_2D(Lx::Int, Ly::Int64, t::Float64)
    L = Lx * Ly
    T = zeros(L, L)

    x = collect(0:L-1) .% Lx       # x positions for sites
    y = div.(collect(0:L-1), Lx)   # y positions for sites
    T_x = (x .+ 1) .% Lx .+ Lx * y      # translation along x-direction
    T_y = x .+ Lx * ((y .+ 1) .% Ly)    # translation along y-direction

    hop_ind_left = [CartesianIndex(i+1, T_x[i+1] + 1) for i in 0 : L-1]
    hop_ind_right = [CartesianIndex(T_x[i+1] + 1, i+1) for i in 0 : L-1]
    hop_ind_down = [CartesianIndex(i+1, T_y[i+1] + 1) for i in 0 : L-1]
    hop_ind_up = [CartesianIndex(T_y[i+1] + 1, i+1) for i in 0 : L-1]

    @views T[hop_ind_left] .= -t
    @views T[hop_ind_right] .= -t
    @views T[hop_ind_down] .= -t
    @views T[hop_ind_up] .= -t
    
    return T
end

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

#######################
##### Test Module #####
#######################

# finite temperature test
@testset "SwapQMC_FT" begin
    ##### Regular 2D Hubbard Model #####

    ### standard sweep ###
    Lx, Ly = 4, 4
    T = hopping_matrix_Hubbard_2D(Lx, Ly, 1.0)

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
    T = hopping_matrix_sshHubbard_1D(L, 1.0, -0.1)
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
    walker::HubbardWalker; ridx::Int = 1
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

    return exp(logr)
end

# ground state test
@testset "SwapQMC_GS" begin
    ##### Attractive 2D Hubbard Model #####
    Lx, Ly = 4, 4
    T = hopping_matrix_Hubbard_2D(Lx, Ly, 1.0)

    system = GenericHubbard(
        # (Nx, Ny), (N_up, N_dn)
        (Lx, Ly, 1), (7, 7),
        # t, U
        T, -4.0,
        # μ
        0.5,
        # β, L
        16.0, 160,
        # data type of the system
        sys_type = Float64,
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

    Θ = div(qmc.K,2)
    θ = div(system.L,2)

    φ₀_up = trial_wf_free(system, 1, T)
    φ₀ = [φ₀_up, copy(φ₀_up)]

    extsys = ExtendedSystem(system, collect(1:8), subsysOrdering=false)
    sampler = EtgSampler(extsys, qmc)

    ### test the regular sweep ###
    walker = HubbardWalker(system, qmc, φ₀)
    # thermalize the walker without saving ratios
    sweep!(system, qmc_nosave, walker, loop_number=10)

    ## Test θ -> 2θ ##
    auxfield = copy(walker.auxfield)
    sweep!_symmetric(system, qmc, walker, collect(Θ+1:2Θ))
    # pick a random point in time [θ:2θ]
    idx_t = rand(θ+1:2θ)
    idx_x = rand(1:system.V)
    idx = (idx_t-θ-1)*system.V + idx_x
    @. auxfield[:, θ+1:idx_t-1] = walker.auxfield[:, θ+1:idx_t-1]
    @. auxfield[1:idx_x-1, idx_t] = walker.auxfield[1:idx_x-1, idx_t]
    r = compute_ratio_regular(system, qmc, auxfield, (idx_x, idx_t), walker.φ₀)
    @test r ≈ walker.tmp_r[idx]

    ## Test 2θ -> Θ ##
    auxfield = copy(walker.auxfield)
    sweep!_symmetric(system, qmc, walker, collect(2Θ:-1:Θ+1))
    # pick a random point in time [θ:2θ]
    idx_t = rand(θ+1:2θ)
    idx_x = rand(1:system.V)
    idx = (2θ-idx_t)*system.V + idx_x + θ*system.V
    @. auxfield[:, idx_t+1:2θ] = walker.auxfield[:, idx_t+1:2θ]
    @. auxfield[1:idx_x-1, idx_t] = walker.auxfield[1:idx_x-1, idx_t]
    r = compute_ratio_regular(system, qmc, auxfield, (idx_x, idx_t), walker.φ₀)
    @test r ≈ walker.tmp_r[idx]

    ## Test Θ -> 0 ##
    auxfield = copy(walker.auxfield)
    sweep!_symmetric(system, qmc, walker, collect(Θ:-1:1))
    # pick a random point in time [1:Θ]
    idx_t = rand(1:θ)
    idx_x = rand(1:system.V)
    idx = (θ-idx_t)*system.V + idx_x + 2*θ*system.V
    @. auxfield[:, θ:-1:idx_t+1] = walker.auxfield[:, θ:-1:idx_t+1]
    @. auxfield[1:idx_x-1, idx_t] = walker.auxfield[1:idx_x-1, idx_t]
    r = compute_ratio_regular(system, qmc, auxfield, (idx_x, idx_t), walker.φ₀)
    @test r ≈ walker.tmp_r[idx]

    ## Test 0 -> Θ ##
    auxfield = copy(walker.auxfield)
    sweep!_symmetric(system, qmc, walker, collect(1:Θ))
    # pick a random point in time [1:Θ]
    idx_t = rand(1:θ)
    idx_x = rand(1:system.V)
    idx = (idx_t-1)*system.V + idx_x + 3*θ*system.V
    @. auxfield[:, 1:idx_t-1] = walker.auxfield[:, 1:idx_t-1]
    @. auxfield[1:idx_x-1, idx_t] = walker.auxfield[1:idx_x-1, idx_t]
    r = compute_ratio_regular(system, qmc, auxfield, (idx_x, idx_t), walker.φ₀)
    @test r ≈ walker.tmp_r[idx]

    ### test the replica sweep ###
    walker1 = HubbardWalker(system, qmc, φ₀)
    walker2 = HubbardWalker(system, qmc, φ₀)
    replica = Replica(extsys, walker1, walker2)

    ### test the first replica in the forward direction ###
    auxfield = copy(walker1.auxfield)
    sweep!_symmetric(system, qmc, replica, walker1, 1, collect(Θ+1:2Θ))

    # pick a random point in time [θ:2θ]
    idx_t = rand(θ+1:2θ)
    idx_x = rand(1:system.V)
    idx = (idx_t-θ-1)*system.V + idx_x
    @. auxfield[:, θ+1:idx_t-1] = walker1.auxfield[:, θ+1:idx_t-1]
    @. auxfield[1:idx_x-1, idx_t] = walker1.auxfield[1:idx_x-1, idx_t]
    r = compute_ratio_replica(extsys, qmc, auxfield, (idx_x, idx_t), walker2, ridx=1)
    @test r ≈ walker1.tmp_r[idx]

    ### test the first replica in the backward direction ###
    auxfield = copy(walker1.auxfield)
    sweep!_symmetric(system, qmc, replica, walker1, 1, collect(Θ:-1:1))
    # pick a random point in time [θ:2θ]
    idx_t = rand(1:θ)
    idx_x = rand(1:system.V)
    idx = (θ - idx_t)*system.V + idx_x + div(system.V*system.L,2)
    @. auxfield[:, idx_t+1:θ] = walker1.auxfield[:, idx_t+1:θ]
    @. auxfield[1:idx_x-1, idx_t] = walker1.auxfield[1:idx_x-1, idx_t]
    r = compute_ratio_replica(extsys, qmc, auxfield, (idx_x, idx_t), walker2, ridx=1)
    @test r ≈ walker1.tmp_r[idx]

    ### switch to the second replica and perform the same test ###
    jump_replica!(replica, 1)

    ## forward direction ##
    auxfield = copy(walker2.auxfield)
    sweep!_symmetric(system, qmc, replica, walker2, 2, collect(Θ+1:2Θ))

    # pick a random point in time [θ:2θ]
    idx_t = rand(θ+1:2θ)
    idx_x = rand(1:system.V)
    idx = (idx_t-θ-1)*system.V + idx_x
    @. auxfield[:, θ+1:idx_t-1] = walker2.auxfield[:, θ+1:idx_t-1]
    @. auxfield[1:idx_x-1, idx_t] = walker2.auxfield[1:idx_x-1, idx_t]
    r = compute_ratio_replica(extsys, qmc, auxfield, (idx_x, idx_t), walker1, ridx=2)
    @test r ≈ walker2.tmp_r[idx]

    ## backward direction ##
    auxfield = copy(walker2.auxfield)
    sweep!_symmetric(system, qmc, replica, walker2, 2, collect(Θ:-1:1))
    # pick a random point in time [θ:2θ]
    idx_t = rand(1:θ)
    idx_x = rand(1:system.V)
    idx = (θ - idx_t)*system.V + idx_x + div(system.V*system.L,2)
    @. auxfield[:, idx_t+1:θ] = walker2.auxfield[:, idx_t+1:θ]
    @. auxfield[1:idx_x-1, idx_t] = walker2.auxfield[1:idx_x-1, idx_t]
    r = compute_ratio_replica(extsys, qmc, auxfield, (idx_x, idx_t), walker1, ridx=2)
    @test r ≈ walker2.tmp_r[idx]

    ### test the sweep with local measurements ###
    # redefine walkers and replica
    walker1 = HubbardWalker(system, qmc, φ₀)
    walker2 = HubbardWalker(system, qmc, φ₀)
    replica = Replica(extsys, walker1, walker2)
    sweep!_symmetric(system, qmc, replica, walker1, sampler, 1, collect(Θ+1:2Θ))
    auxfield = copy(walker1.auxfield)
    sweep!_symmetric(system, qmc, replica, walker1, sampler, 1, collect(Θ:-1:1))
    # pick a random point in time [θ:2θ]
    idx_t = rand(1:θ)
    idx_x = rand(1:system.V)
    idx = (θ - idx_t)*system.V + idx_x + div(system.V*system.L,2)
    @. auxfield[:, idx_t+1:θ] = walker1.auxfield[:, idx_t+1:θ]
    @. auxfield[1:idx_x-1, idx_t] = walker1.auxfield[1:idx_x-1, idx_t]
    r = compute_ratio_replica(extsys, qmc, auxfield, (idx_x, idx_t), walker2, ridx=1)
    @test r ≈ walker1.tmp_r[idx]

    jump_replica!(replica, 1)
    sweep!_symmetric(system, qmc, replica, walker2, sampler, 2, collect(Θ+1:2Θ))
    auxfield = copy(walker2.auxfield)
    sweep!_symmetric(system, qmc, replica, walker2, sampler, 2, collect(Θ:-1:1))
    # pick a random point in time [θ:2θ]
    idx_t = rand(1:θ)
    idx_x = rand(1:system.V)
    idx = (θ - idx_t)*system.V + idx_x + div(system.V*system.L,2)
    @. auxfield[:, idx_t+1:θ] = walker2.auxfield[:, idx_t+1:θ]
    @. auxfield[1:idx_x-1, idx_t] = walker2.auxfield[1:idx_x-1, idx_t]
    r = compute_ratio_replica(extsys, qmc, auxfield, (idx_x, idx_t), walker1, ridx=2)
    @test r ≈ walker2.tmp_r[idx]
end

# subsystem sampling test
@testset "SwapQMC_Subsys" begin
    ##### Attractive 2D Hubbard Model #####
    Lx, Ly = 4, 4
    T = hopping_matrix_Hubbard_2D(Lx, Ly, 1.0)

    system = GenericHubbard(
        # (Nx, Ny), (N_up, N_dn)
        (Lx, Ly, 1), (7, 7),
        # t, U
        T, -0.5,
        # μ
        0.0,
        # β, L
        1.0, 20,
        # data type of the system
        sys_type = Float64,
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

    Θ = div(qmc.K,2)
    θ = div(system.L,2)

    φ₀_up = trial_wf_free(system, 1, T)
    φ₀ = [φ₀_up, copy(φ₀_up)]

    extsys = ExtendedSystem(system, collect(1:4), subsysOrdering=false)
    sampler = EtgSampler(extsys, qmc)
    walker = HubbardSubsysWalker(extsys, qmc, φ₀)

    ### test the forward direction ###
    auxfield = copy(walker.auxfield)
    sweep!_symmetric(system, qmc, walker, collect(Θ+1:2Θ))

    # pick a random point in time [θ:2θ]
    idx_t = rand(θ+1:2θ)
    idx_x = rand(1:system.V)
    idx = (idx_t-θ-1)*system.V + idx_x
    @. auxfield[:, θ+1:idx_t-1] = walker.auxfield[:, θ+1:idx_t-1]
    @. auxfield[1:idx_x-1, idx_t] = walker.auxfield[1:idx_x-1, idx_t]
    # then create new test walkers
    walker′ = HubbardSubsysWalker(extsys, qmc, φ₀, auxfield=auxfield)
    auxfield[idx_x, idx_t] *= -1
    walker″ = HubbardSubsysWalker(extsys, qmc, φ₀, auxfield=auxfield)

    r = det(walker″.ImGA⁻¹[1]) / det(walker′.ImGA⁻¹[1])
    @test r^2 ≈ walker.tmp_r[idx]

    ### test the backward direction ###
    auxfield = copy(walker.auxfield)
    sweep!_symmetric(system, qmc, walker, collect(Θ:-1:1))
    # pick a random point in time [θ:2θ]
    idx_t = rand(1:θ)
    idx_x = rand(1:system.V)
    idx = (θ - idx_t)*system.V + idx_x + div(system.V*system.L,2)
    @. auxfield[:, idx_t+1:θ] = walker.auxfield[:, idx_t+1:θ]
    @. auxfield[1:idx_x-1, idx_t] = walker.auxfield[1:idx_x-1, idx_t]
    # then create new test walkers
    walker′ = HubbardSubsysWalker(extsys, qmc, φ₀, auxfield=auxfield)
    auxfield[idx_x, idx_t] *= -1
    walker″ = HubbardSubsysWalker(extsys, qmc, φ₀, auxfield=auxfield)

    r = det(walker″.ImGA⁻¹[1]) / det(walker′.ImGA⁻¹[1])
    @test r^2 ≈ walker.tmp_r[idx]

    ### test the sweep with local measurements ###
    walker = HubbardSubsysWalker(extsys, qmc, φ₀)
    sweep!_symmetric(system, qmc, walker, collect(Θ+1:2Θ))
    auxfield = copy(walker.auxfield)
    sweep!_symmetric(system, qmc, walker, sampler, collect(Θ:-1:1))
    # pick a random point in time [θ:2θ]
    idx_t = rand(1:θ)
    idx_x = rand(1:system.V)
    idx = (θ - idx_t)*system.V + idx_x + div(system.V*system.L,2)
    @. auxfield[:, idx_t+1:θ] = walker.auxfield[:, idx_t+1:θ]
    @. auxfield[1:idx_x-1, idx_t] = walker.auxfield[1:idx_x-1, idx_t]
    # then create new test walkers
    walker′ = HubbardSubsysWalker(extsys, qmc, φ₀, auxfield=auxfield)
    auxfield[idx_x, idx_t] *= -1
    walker″ = HubbardSubsysWalker(extsys, qmc, φ₀, auxfield=auxfield)

    r = det(walker″.ImGA⁻¹[1]) / det(walker′.ImGA⁻¹[1])
    @test r^2 ≈ walker.tmp_r[idx]
end
