using SwapQMC, JLD

function run_incremental_sampling_gs(
    extsys::ExtendedSystem{Sys}, qmc::QMC, φ₀::Vector{Wf}, λₖ::Float64, Nₖ::Int,
    path::String, filename::String
) where {Sys<:Hubbard, Wf<:AbstractMatrix}

    system = extsys.system

    walker1 = HubbardWalker(system, qmc, φ₀)
    walker2 = HubbardWalker(system, qmc, φ₀)

    replica = Replica(extsys, walker1, walker2, λₖ=λₖ)

    bins = qmc.measure_interval

    detgA = zeros(Float64, qmc.nsamples)

    # warm-up steps
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

    # measurements
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

        detgA[i] = exp(-2 * replica.logdetGA[] / Nₖ)
    end
    
    # store the measurement
    jldopen("$(path)/$(filename)", "w") do file
        write(file, "detgA", detgA)
    end

    return nothing
end
