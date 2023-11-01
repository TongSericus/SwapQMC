module SwapQMC
    using Reexport: @reexport
    @reexport using LinearAlgebra, Statistics, Random

    # import packages from https://github.com/cohensbw
    @reexport using StableLinearAlgebra, LatticeUtilities, Checkerboard

    import StableLinearAlgebra as Slinalg
    import StableLinearAlgebra: mul!, lmul!, rmul!, rdiv!
    
    # system and simulation related quantities, linalg operations
    export System, ExtendedSystem, QMC
    export Hubbard, GenericHubbard
    include("./base/base_utils.jl")
    include("./base/systems.jl")
    include("./base/qmc_variable.jl")
    include("./base/linalg_RDM.jl")
    include("./base/linalg_SVD.jl")
    include("./base/matrix_generator.jl")
    export hopping_matrix_Hubbard_1d, hopping_matrix_ssh_1d, 
           hopping_matrix_Hubbard_2d, hopping_matrix_ssh_2d
    include("./base/one_body_matrices.jl")
    
    # random walker and replica at finite and zero temperature
    export HubbardGCWalker, HubbardGCSwapper, Replica
    export trial_wf_free, trial_wf_HF, trial_wf_bcs
    export HubbardWalker
    include("./walker.jl")
    include("./trial_wavefunctions.jl")
    include("./walker_gs.jl")
    include("./replica.jl")

    # Sampler
    export EtgData, EtgSampler
    include("./base/sampler.jl")

    # MC operations
    include("./operations.jl")
    include("./operations_gs.jl")
    
    # MC propagations
    export sweep!
    include("./propagation/standard.jl")
    include("./propagation/swap.jl")
    include("./propagation/replica.jl")
    export sweep!_symmetric, jump_replica!
    include("./propagation/standard_gs.jl")
    include("./propagation/replica_gs.jl")
    
    export TunableHubbard, TunableHubbardWalker,
           MuTuner, dynamical_tuning
    include("./dynamical_tuning/base.jl")
    include("./dynamical_tuning/dynamical_tuning.jl")

    export measure_expS2!, measure_Pn!, measure_Pn2!
    include("./measurements.jl")
end