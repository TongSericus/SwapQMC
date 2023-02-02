using SwapQMC
using SwapQMC: Cluster, update_cluster!, run_full_propagation, inv_IpμA!, expand!
using Test

@testset "SwapQMC.jl" begin

    ### Testing regular sweep ###
    # Initialize model parameters
    system = IonicHubbard(
        # (Nx, Ny), (N_up, N_dn)
        (4, 4), (8, 8),
        # t, U, Δ
        1.0, 2.0, 0.0,
        # μ
        1.0,
        # β, L
        6.0, 60,
        useFirstOrderTrotter = true
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

    # update the first propagator matrix cluster
    update_cluster!(walker, system, qmc, 1)

    # create the test matrix cluster
    K = qmc.K
    Bc = walker.Bc.B
    Bc_test = Cluster(B = vcat(Bc[2 : K], [Bc[1]], Bc[K+2 : 2*K], [Bc[K + 1]]))

    # propagate using Bc_test
    F = run_full_propagation(Bc_test, walker.ws)
    G = Matrix{Float64}(1.0I, system.V, system.V)
    # testing G
    inv_IpA!(G, F[1], walker.ws)
    @test walker.G[1] ≈ G
    inv_IpA!(G, F[2], walker.ws)
    @test walker.G[2] ≈ G

    ### Testing swap sweep ###
    # create a bipartition
    Aidx = collect(1:8)
    extsys = ExtendedSystem(system, Aidx)

    # create replica walkers
    walker = HubbardGCWalker(system, qmc)
    walker′ = HubbardGCWalker(system, qmc)

    # create a swapper
    swapper = HubbardGCSwapper(extsys, walker, walker′)

    # update the first propagator matrix cluster of the first walker
    update_cluster!(walker, swapper, extsys, qmc, 1, 1)

    # create the temporal swapper
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

    # testing G
    G = Matrix{Float64}(1.0I, extsys.Vext, extsys.Vext)
    inv_IpA!(G, F[1], swapper.ws)
    @test G ≈ swapper.G[1]
    inv_IpA!(G, F[2], swapper.ws)
    @test G ≈ swapper.G[2]
end
