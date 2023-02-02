struct QMC
    """
        All static parameters needed for QMC.
        
        nwarmups -> number of warm-up steps
        nsamples -> number of samples to collect
        measure_interval -> number of sweeps between two samples
        isCP -> Use Column-Pivoting QR decomposition
        stab_interval -> number of matrix multiplications before the stablization
        update_interval -> number of steps after which the population control and calibration is required
        isLowrank -> if enabling the low-rank truncation
        lrThld -> low-rank truncation threshold (from above)
    """
    ### MCMC (Metropolis) ###
    nwarmups::Int64             # number of warm-up steps
    nsamples::Int64             # number of samples to collect
    measure_interval::Int64     # number of sweeps between two samples
    ### Numerical Stablization ###
    stab_interval::Int64        # number of matrix multiplications before the stablization
    K::Int64                    # number of groups of matrices
    K_interval::Vector{Int64}
    update_interval::Int64      # number of steps after which a calibration is required

    function QMC(
        system::System, 
        nwarmups::Int64, nsamples::Int64, measure_interval::Int64, 
        stab_interval::Int64, update_interval::Int64
    )
        # number of clusters
        system.L % stab_interval == 0 ? K = div(system.L, stab_interval) : K = div(system.L, stab_interval) + 1
        # group the rest of the matrices as the last cluster
        Le = mod(system.L, stab_interval)
        K_interval = [stab_interval for _ in 1 : K]
        Le == 0 || (K_interval[end] = Le)
    
        return new(
            nwarmups, nsamples, measure_interval,
            stab_interval, K, K_interval, update_interval
        )
    end
end
