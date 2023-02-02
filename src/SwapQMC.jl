module SwapQMC
    using Reexport: @reexport
    @reexport using LinearAlgebra, StableLinearAlgebra, Statistics, Random, FFTW

    import StableLinearAlgebra as Slinalg
    import StableLinearAlgebra: mul!, lmul!, rmul!, rdiv!

    export System, ExtendedSystem, QMC
    export Hubbard, BilayerHubbard, IonicHubbard
    include("./base/systems.jl")
    include("./base/qmc_variable.jl")
    include("./base/linalg_RDM.jl")
    include("./base/matrix_generator.jl")

    export HubbardGCWalker, HubbardGCSwapper,
           sweep!
    include("./walker.jl")
    include("./operations.jl")
    include("./propagation.jl")

    export TunableHubbard, TunableHubbardWalker,
           MuTuner, dynamical_tuning
    include("./dynamical_tuning/base.jl")
    include("./dynamical_tuning/dynamical_tuning.jl")

    export EtgData, EtgMeasurement, measure_EE, measure_EE!
    include("./measurements.jl")
end