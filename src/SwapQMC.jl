module SwapQMC
    using Reexport: @reexport
    @reexport using LinearAlgebra, StableLinearAlgebra, Statistics, Random, FFTW

    import StableLinearAlgebra as Slinalg
    import StableLinearAlgebra: mul!, lmul!, rmul!, rdiv!

    export BilayerHubbard, IonicHubbard, ExtendedSystem, QMC
    include("./base/systems.jl")
    include("./base/qmc_variable.jl")
    include("./base/linalg_RDM.jl")
    include("./base/matrix_generator.jl")

    export HubbardGCWalker, HubbardGCSwapper,
           sweep!
    include("./walker.jl")
    include("./operations.jl")
    include("./propagation.jl")

    export EtgData, EtgMeasurement, measure_EE, measure_EE!
    include("./measurements.jl")
end