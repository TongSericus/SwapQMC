using SwapQMC, JLD

function run_swap_sampling_gs(
    extsys::ExtendedSystem{Sys}, qmc::QMC, φ₀::Vector{Wf},
    direction::Int, path::String, filename::String
) where {Sys<:Hubbard, Wf<:AbstractMatrix}

    system = extsys.system

    walker1 = HubbardWalker(system, qmc, φ₀)
    walker2 = HubbardWalker(system, qmc, φ₀)

    replica = Replica(extsys, walker1, walker2)

    bins = qmc.measure_interval
    half_bins = div(qmc.measure_interval, 2)

    # measurements
    sampler = EtgSampler(extsys, qmc)

    if direction == 1   # sample Z² space
        for i in 1 : qmc.nwarmups
            if (i-1) % 256 < 128
                sweep!(system, qmc, walker1, loop_number=1)
            else
                sweep!(system, qmc, walker2, loop_number=1)
            end
        end

        for i in 1 : qmc.nsamples
            if (i-1) % 256 < 128
                sweep!(system, qmc, walker1, replica, loop_number=half_bins)
            else
                sweep!(system, qmc, walker2, replica, loop_number=half_bins)
            end

            measure_replica!(sampler, replica, direction=1)
        end
    
        # store the measurement
        jldopen("$(path)/$(filename)", "w") do file
        write(file, "p_num", sampler.p)
    end
    else                # sample Zₐ² space
        for i in 1 : qmc.nwarmups
            if (i-1) % 256 < 127
                sweep!(system, qmc, replica, walker1, 1, loop_number=1, jumpReplica=false)
            elseif (i-1) % 256 == 127
                sweep!(system, qmc, replica, walker1, 1, loop_number=1, jumpReplica=true)
            elseif 127 < (i-1) % 256 < 255
                sweep!(system, qmc, replica, walker2, 2, loop_number=1, jumpReplica=false)
            else
                sweep!(system, qmc, replica, walker2, 2, loop_number=1, jumpReplica=true)
            end
        end

        for i in 1 : qmc.nsamples
            if (i-1) % 256 < 127
                sweep!(system, qmc, replica, walker1, 1, loop_number=bins, jumpReplica=false)
            elseif (i-1) % 256 == 127
                sweep!(system, qmc, replica, walker1, 1, loop_number=bins, jumpReplica=true)
            elseif 127 < (i-1) % 256 < 255
                sweep!(system, qmc, replica, walker2, 2, loop_number=bins, jumpReplica=false)
            else
                sweep!(system, qmc, replica, walker2, 2, loop_number=bins, jumpReplica=true)
            end

            measure_replica!(sampler, replica, direction=2)
        end

        # store the measurement
        jldopen("$(path)/$(filename)", "w") do file
            write(file, "p_denom", sampler.p)
            write(file, "Pn2_up", sampler.Pn₊)
            write(file, "Pn2_dn", sampler.Pn₋)
        end
    end

    return nothing
end
