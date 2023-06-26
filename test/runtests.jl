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

function compute_gs_projected_matrix(walker)
    B_bra = SwapQMC.lmul!_svd(walker.Ul, walker.φ₀T[1], walker.Fl[1])
    B_ket = SwapQMC.rmul!_svd(walker.Ur, walker.Fr[1], walker.φ₀[1])
    B1 = B_bra[1] * Diagonal(B_bra[2]) * B_bra[3]
    B2 = B_ket[1] * Diagonal(B_ket[2]) * B_ket[3]
    return det(B1*B2)
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

    walker1 = HubbardWalker(system, qmc, φ₀)
    walker2 = HubbardWalker(system, qmc, φ₀)

    extsys = ExtendedSystem(system, collect(1:8), subsysOrdering=false)
    replica = Replica(extsys, walker1, walker2)

    ### test the first replica in the forward direction ###
    auxfield = copy(walker1.auxfield)
    #sweep!(system, qmc, replica, walker1, 1)
    sweep!_symmetric(system, qmc, replica, walker1, 1, collect(Θ+1:2Θ))

    # pick a random point in time [θ:2θ]
    idx_t = rand(θ+1:2θ)
    idx_x = rand(1:system.V)
    idx = (idx_t-θ-1)*system.V + idx_x
    @. auxfield[:, θ+1:idx_t-1] = walker1.auxfield[:, θ+1:idx_t-1]
    @. auxfield[1:idx_x-1, idx_t] = walker1.auxfield[1:idx_x-1, idx_t]
    # then create new test walkers
    walker3 = HubbardWalker(system, qmc, φ₀, auxfield=auxfield)
    auxfield[idx_x, idx_t] *= -1
    walker4 = HubbardWalker(system, qmc, φ₀, auxfield=auxfield)
    # and test replicas
    replica2 = Replica(extsys, walker3, walker2)
    replica3 = Replica(extsys, walker4, walker2)

    det_B_old = compute_gs_projected_matrix(walker3)
    det_B_new = compute_gs_projected_matrix(walker4)
    r = abs((det_B_new / det_B_old)^2) * exp(2*replica2.logdetGA[] - 2*replica3.logdetGA[])
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
    # then create new test walkers
    walker3 = HubbardWalker(system, qmc, φ₀, auxfield=auxfield)
    auxfield[idx_x, idx_t] *= -1
    walker4 = HubbardWalker(system, qmc, φ₀, auxfield=auxfield)
    # and test replicas
    replica2 = Replica(extsys, walker3, walker2)
    replica3 = Replica(extsys, walker4, walker2)

    B_old = compute_gs_projected_matrix(walker3)
    B_new = compute_gs_projected_matrix(walker4)
    r = abs((det(B_new) / det(B_old))^2) * exp(2*replica2.logdetGA[] - 2*replica3.logdetGA[])
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
    # then create new test walkers
    walker3 = HubbardWalker(system, qmc, φ₀, auxfield=auxfield)
    auxfield[idx_x, idx_t] *= -1
    walker4 = HubbardWalker(system, qmc, φ₀, auxfield=auxfield)
    # and test replicas
    replica2 = Replica(extsys, walker1, walker3)
    replica3 = Replica(extsys, walker1, walker4)

    det_B_old = compute_gs_projected_matrix(walker3)
    det_B_new = compute_gs_projected_matrix(walker4)
    r = abs((det_B_new / det_B_old)^2) * exp(2*replica2.logdetGA[] - 2*replica3.logdetGA[])
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
    # then create new test walkers
    walker3 = HubbardWalker(system, qmc, φ₀, auxfield=auxfield)
    auxfield[idx_x, idx_t] *= -1
    walker4 = HubbardWalker(system, qmc, φ₀, auxfield=auxfield)
    # and test replicas
    replica2 = Replica(extsys, walker1, walker3)
    replica3 = Replica(extsys, walker1, walker4)

    B_old = compute_gs_projected_matrix(walker3)
    B_new = compute_gs_projected_matrix(walker4)
    r = abs((det(B_new) / det(B_old))^2) * exp(2*replica2.logdetGA[] - 2*replica3.logdetGA[])
    @test r ≈ walker2.tmp_r[idx]
end