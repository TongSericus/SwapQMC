using SwapQMC
using SwapQMC: Cluster, update_cluster!, run_full_propagation, inv_IpμA!, expand!
using Test

function test_regular_sweep(
    system::System, qmc::QMC, walker;
    T::DataType = eltype(system.auxfield)
)
    # update the first propagator matrix cluster
    update_cluster!(walker, system, qmc, 1)

    # create the test matrix cluster
    K = qmc.K
    Bc = walker.Bc.B
    Bc_test = Cluster(B = vcat(Bc[2 : K], [Bc[1]], Bc[K+2 : 2*K], [Bc[K + 1]]))

    # propagate using Bc_test
    F = run_full_propagation(Bc_test, walker.ws)
    V = system.V
    G = [Matrix{T}(1.0I, V, V), Matrix{T}(1.0I, V, V)]

    # compute G
    inv_IpA!(G[1], F[1], walker.ws)
    inv_IpA!(G[2], F[2], walker.ws)

    return G
end

function test_swap_sweep(
    extsys::ExtendedSystem, qmc::QMC, walker, swapper; 
    T::DataType = eltype(extsys.system.auxfield)
)
    K = qmc.K

    # update the first propagator matrix cluster of the first walker
    update_cluster!(walker, swapper, extsys, qmc, 1, 1)

    # create the temporal decomposition
    F = swapper.C
    L⁺ = swapper.L
    R⁺ = swapper.R
    Bc = walker.Bc.B
    L = walker.FC.B
    expand!(R⁺, ldr(Bc[1], walker.ws), 1)
    expand!(L⁺, L[1], 1)
    lmul!(R⁺, F[1], swapper.ws)
    rmul!(F[1], L⁺, swapper.ws)
    expand!(R⁺, ldr(Bc[K + 1], walker.ws), 1)
    expand!(L⁺, L[K + 1], 1)
    lmul!(R⁺, F[2], swapper.ws)
    rmul!(F[2], L⁺, swapper.ws)

    # compute G
    V = extsys.Vext
    G = [Matrix{T}(1.0I, V, V), Matrix{T}(1.0I, V, V)]
    inv_IpA!(G[1], F[1], swapper.ws)
    inv_IpA!(G[2], F[2], swapper.ws)

    return G
end

@testset "SwapQMC.jl" begin

    ### Testing Ionic Hubbard model ###
    ## Regular sweep ##
    # Initialize model parameters
    system = IonicHubbard(
        # (Nx, Ny), (N_up, N_dn)
        (4, 4), (8, 8),
        # t, U, Δ
        1.0, 2.0, 0.0,
        # μ
        1.0,
        # β, L
        6.0, 60
    )
    qmc = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        5, Int64(5), 5,
        # stablization and update interval
        5, 5
    )

    # create a random walker
    walker = HubbardGCWalker(system, qmc)

    # propagate through first cluster
    G = test_regular_sweep(system, qmc, walker)

    # testing G
    @test walker.G[1] ≈ G[1]
    @test walker.G[2] ≈ G[2]

    ## Swap sweep ##
    # create a bipartition
    Aidx = collect(1:8)
    extsys = ExtendedSystem(system, Aidx)

    # create replica walkers
    walker = HubbardGCWalker(system, qmc)
    walker′ = HubbardGCWalker(system, qmc)

    # create a swapper
    swapper = HubbardGCSwapper(extsys, walker, walker′)

    # propagate through first cluster of the first walker
    G = test_swap_sweep(extsys, qmc, walker, swapper)

    # testing G
    @test G[1] ≈ swapper.G[1]
    @test G[2] ≈ swapper.G[2]

    ### Testing Bilayer Hubbard model ###
    # Same as above
    system = BilayerHubbard(
        # (Nx, Ny), (N_up, N_dn)
        (4, 4), (8, 8),
        # t, t′, U
        1.0, 1.0, 4.0,
        # μ
        2.0,
        # β, L
        6.0, 60
    )
    qmc = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        5, Int64(5), 5,
        # stablization and update interval
        5, 5
    )

    walker = HubbardGCWalker(system, qmc)
    G = test_regular_sweep(system, qmc, walker)

    # testing G
    @test walker.G[1] ≈ G[1]
    @test walker.G[2] ≈ G[2]

    ## Swap sweep ##
    # create a bipartition
    Aidx = collect(1:8)
    extsys = ExtendedSystem(system, Aidx)

    walker = HubbardGCWalker(system, qmc)
    walker′ = HubbardGCWalker(system, qmc)

    swapper = HubbardGCSwapper(extsys, walker, walker′)

    G = test_swap_sweep(extsys, qmc, walker, swapper)

    # testing G
    @test G[1] ≈ swapper.G[1]
    @test G[2] ≈ swapper.G[2]

    ### Testing Bilayer Hubbard model with complex HS transform ###
    # Same as above
    system = BilayerHubbard(
        # (Nx, Ny), (N_up, N_dn)
        (4, 4), (8, 8),
        # t, t′, U
        1.0, 1.0, 4.0,
        # μ
        2.0,
        # β, L
        6.0, 60,
        # auxiliary field is coupled to charge
        useComplexHST = true
    )
    qmc = QMC(
        system,
        # number of warm-ups, samples and measurement interval
        5, Int64(5), 5,
        # stablization and update interval
        5, 5
    )

    walker = HubbardGCWalker(system, qmc)
    G = test_regular_sweep(system, qmc, walker)

    # testing G
    @test walker.G[1] ≈ G[1]

    ## Swap sweep ##
    # create a bipartition
    Aidx = collect(1:8)
    extsys = ExtendedSystem(system, Aidx)

    walker = HubbardGCWalker(system, qmc)
    walker′ = HubbardGCWalker(system, qmc)

    swapper = HubbardGCSwapper(extsys, walker, walker′)

    G = test_swap_sweep(extsys, qmc, walker, swapper)

    # testing G
    @test G[1] ≈ swapper.G[1]
end
