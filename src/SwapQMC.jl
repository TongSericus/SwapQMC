module SwapQMC
    using Reexport: @reexport
    @reexport using LinearAlgebra, Statistics, Random

    # import packages from https://github.com/cohensbw
    @reexport using StableLinearAlgebra, LatticeUtilities, Checkerboard

    import StableLinearAlgebra as Slinalg
    import StableLinearAlgebra: mul!, lmul!, rmul!, rdiv!

    export System, ExtendedSystem, QMC
    export Hubbard, GenericHubbard
    include("./base/systems.jl")
    include("./base/qmc_variable.jl")
    include("./base/linalg_RDM.jl")
    include("./base/linalg_SVD.jl")
    include("./base/matrix_generator.jl")
    export hopping_matrix_Hubbard_1d, hopping_matrix_ssh_1d, 
           hopping_matrix_Hubbard_2d, hopping_matrix_ssh_2d
    include("./base/one_body_matrices.jl")

    export HubbardGCWalker, HubbardGCSwapper, Replica
    export sweep!
    include("./walker.jl")
    include("./replica.jl")
    include("./operations.jl")
    include("./propagation/standard.jl")
    include("./propagation/swap.jl")
    include("./propagation/replica.jl")

    # ground state
    export HubbardWalker
    export trial_wf_free
    export sweep!_symmetric, jump_replica!
    include("./trial_wavefunctions.jl")
    include("./walker_gs.jl")
    include("./operations_gs.jl")
    include("./propagation/standard_gs.jl")
    include("./propagation/replica_gs.jl")

    # subsystem sampling
    export HubbardSubsysWalker
    include("operations_subsys.jl")
    include("./propagation/standard_subsys.jl")

    export TunableHubbard, TunableHubbardWalker,
           MuTuner, dynamical_tuning
    include("./dynamical_tuning/base.jl")
    include("./dynamical_tuning/dynamical_tuning.jl")

    export EtgData, EtgMeasurement, 
           measure_EE, measure_EE!, measure_Pn!
    include("./measurements.jl")
end