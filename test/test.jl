using SwapQMC

system = BilayerHubbard(
    (4, 4), (8, 8),
    # t, t′, U
    1.0, 2.0, 4.0,
    # μ
    0.0,
    # β, L
    10.0, 100,
    useComplexHST = true
)

qmc = QMC(
    system,
    # number of warm-ups, samples and measurement interval
    50, Int64(1e3), 20,
    # stablization and update interval
    5, 5
)

Aidx = collect(1:2:31)
extsys = ExtendedSystem(system, Aidx)

Random.seed!(1234)
σ1 = rand([-1, 1], system.V, system.L)
σ2 = rand([-1, 1], system.V, system.L)

walker1 = HubbardGCWalker(system, qmc, auxfield = copy(σ1))
walker2 = HubbardGCWalker(system, qmc, auxfield = copy(σ2))

swapper = HubbardGCSwapper(extsys, walker1, walker2)

#i = SwapQMC.flip_HSField.(walker1.auxfield[1, 1])
#G = copy(swapper1.G)
#SwapQMC.wrap_G!(G[1], swapper1.Bk[1], swapper1.Bk⁻¹[1], swapper1.ws)
#G2 = copy(swapper2.G)
#SwapQMC.wrap_G!(G2[1], swapper2.Bk[1], swapper2.Bk⁻¹[1], swapper2.ws)
#r, d_up, d_dn = SwapQMC.compute_Metropolis_ratio(G, walker1.α, i, 1, isComplex = true)
#
#SwapQMC.update_G!(G[1], walker1.α[1, i], d_up, 1, swapper1.ws)
#
#println(i)
#println(r ≈ exp(sum(swapper2.weight .- swapper1.weight)))

#G = copy(walker1.G)
#SwapQMC.wrap_G!(G[1], system.Bk, system.Bk⁻¹, walker1.ws)
#G2 = copy(walker3.G)
#SwapQMC.wrap_G!(G2[1], system.Bk, system.Bk⁻¹, walker3.ws)
#
#i = SwapQMC.flip_HSField.(walker1.auxfield[1, 1])
#r, d_up, d_dn = SwapQMC.compute_Metropolis_ratio(G, walker1.α, i, 1, isComplex = true)
#
#println(i)
#println(r ≈ exp(sum(walker3.weight .- walker1.weight)))

#γ = walker.α[1, i]
#d1 = (1 + γ * (1 - G[1][1, 1]))^2 / (γ + 1)

#u = -G[1][:, 1]
#u[1] += 1
#d = 1 + γ*u[1]
#SwapQMC.update_G!(G[1], γ, d, 1, walker.ws)