using SwapQMC, JLD

"""
    MC simulation for entanglement measurements
"""
function run_swap_gce(
    extsys::ExtendedSystem, qmc::QMC, 
    direction::Int64, path::String, filename::String
)
    system = extsys.system
    T = system.useComplexHST ? ComplexF64 : Float64

    ### Sweep in the Z_{A, 2} space ###
    if direction == 1

        etgdata = EtgData(extsys)
        etgm = EtgMeasurement(extsys)

        walker1 = HubbardGCWalker(system, qmc)
        walker2 = HubbardGCWalker(system, qmc)
        replica = Replica(extsys, walker1, walker2)

        p = zeros(Float64, qmc.nsamples)
        Pn2_up = Matrix{T}(1.0I, extsys.LA+1, qmc.nsamples)
        Pn2_dn = Matrix{T}(1.0I, extsys.LA+1, qmc.nsamples)

        for i in 1 : qmc.nwarmups
            # sweep over the first replica
            sweep!(system, qmc, replica, walker1, 1)
            # sweep over the second replica
            sweep!(system, qmc, replica, walker2, 2)
        end

        for i in 1 : qmc.nsamples

            for j in 1 : qmc.measure_interval
                # sweep over the first replica
                sweep!(system, qmc, replica, walker1, 1)
                # sweep over the second replica
                sweep!(system, qmc, replica, walker2, 2)
            end
            
            # measurement
            measure_EE!(etgm, etgdata, extsys, replica)

            # store the measurements
            p[i] = etgm.p[]
            @views copyto!(Pn2_up[:, i], etgm.Pn2₊)
            @views copyto!(Pn2_dn[:, i], etgm.Pn2₋)
	    end
	
        jldopen("$(path)/$(filename)", "w") do file
            write(file, "p", p)
            write(file, "Pn2_up", Pn2_up)
            write(file, "Pn2_dn", Pn2_dn)
        end

    ### Sweep in the Z² space ###
    elseif direction == 2

        walker1 = HubbardGCWalker(system, qmc)
        walker2 = HubbardGCWalker(system, qmc)
        replica = Replica(extsys, walker1, walker2)

        p = zeros(Float64, qmc.nsamples)
        sgn = zeros(T, qmc.nsamples)
                                                                                                                                          
        for i in 1 : qmc.nwarmups
            sweep!(system, qmc, walker1)
            sweep!(system, qmc, walker2)
        end
                                                                                                                                          
        for i in 1 : qmc.nsamples
                                                                                                                                          
            for j in 1 : qmc.measure_interval
                sweep!(system, qmc, walker1)
                sweep!(system, qmc, walker2)
            end

            update!(replica)
            p[i] = exp(-2 * replica.logdetGA[])
            sgn[i] = prod(walker1.sign) * prod(walker2.sign)
        end

        jldopen("$(path)/$(filename)", "w") do file
            write(file, "sgn", sgn)
            write(file, "p", p)
        end
    end
end
