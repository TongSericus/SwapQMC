using SwapQMC

const U = 0.2
const μ = 0.10080122704425552

const system1 = IonicHubbard(
    (6, 6), (18, 18),
    # t, U, Δ
    1.0, U, 0.5,
    # μ
    μ,
    # β, L
    12.0, 240
)

const qmc1 = QMC(
    system1,
    # number of warm-ups, samples and measurement interval
    5, Int64(5), 2,
    # stablization and update interval
    5, 5
)

const system2 = BilayerHubbard(
    (6, 3), (16, 16),
    # t, t′, U
    1.0, 2.0, 2.0,
    # μ
    1.0,
    # β, L
    12.0, 240
)

const qmc2 = QMC(
    system2,
    # number of warm-ups, samples and measurement interval
    5, Int64(5), 2,
    # stablization and update interval
    5, 5
)

const Aidx = collect(1:8)
const extsys1 = ExtendedSystem(system1, Aidx)
const extsys2 = ExtendedSystem(system2, Aidx)

etgdata1 = EtgData(extsys1)
etgm1 = EtgMeasurement(extsys1)

etgdata2 = EtgData(extsys2)
etgm2 = EtgMeasurement(extsys2)

walker1_1 = HubbardGCWalker(system1, qmc1)
walker1_2 = HubbardGCWalker(system1, qmc1)
swapper1 = HubbardGCSwapper(extsys1, walker1_1, walker1_2)

walker2_1 = HubbardGCWalker(system2, qmc2)
walker2_2 = HubbardGCWalker(system2, qmc2)
swapper2 = HubbardGCSwapper(extsys2, walker2_1, walker2_2)

