struct QMC
    """
        Static parameters for QMC
    """
    ### MCMC (Metropolis) ###
    nwarmups::Int64             # number of warm-up steps
    nsamples::Int64             # number of samples to collect
    measure_interval::Int64     # number of sweeps between two samples
    ### Extra Control Parameters ###
    useHeatbath::Bool           # use standard Metropolis ratio or the heat-bath ratio
    forceSymmetry::Bool         # force the symmetry between spin-up and spin-down channel
    ### Numerical Stablization ###
    stab_interval::Int64        # number of matrix multiplications before the stablization
    K::Int64                    # number of groups of matrices
    K_interval::Vector{Int64}   # collection of K
    update_interval::Int64      # number of steps after which a calibration is required
    ### Flags for debugging purpose ###
    saveRatio::Bool

    function QMC(
        system::System, 
        nwarmups::Int64, nsamples::Int64, measure_interval::Int64, 
        stab_interval::Int64, update_interval::Int64;
        useHeatbath::Bool = true, forceSymmetry::Bool = false,
        saveRatio::Bool = false
    )
        # number of clusters
        system.L % stab_interval == 0 ? K = div(system.L, stab_interval) : K = div(system.L, stab_interval) + 1
        # group the rest of the matrices as the last cluster
        Le = mod(system.L, stab_interval)
        K_interval = [stab_interval for _ in 1 : K]
        Le == 0 || (K_interval[end] = Le)
    
        return new(
            nwarmups, nsamples, measure_interval,
            useHeatbath, forceSymmetry,
            stab_interval, K, K_interval, update_interval,
            saveRatio
        )
    end
end
