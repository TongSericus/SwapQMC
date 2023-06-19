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

@testset "SwapQMC.jl" begin
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