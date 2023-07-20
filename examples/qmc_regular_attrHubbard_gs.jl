using SwapQMC, JLD

# Regular ground state samping for measuring spin-spin correlations for negative-U Hubbard model
function run_regular_sampling_gs(
    system::System, qmc::QMC, φ₀::Vector{Wf},
    path::String, filename::String
) where {Wf<:AbstractMatrix}

    walker = HubbardWalker(system, qmc, φ₀)
    half_bins = div(qmc.measure_interval,2)

    # density matrix
    ρ₊ = DensityMatrix(walker.G[1])
    ρ₋ = DensityMatrix(walker.G[2])
    # measurements
    sampler = CorrFuncSampler(system, qmc)

    sweep!(system, qmc, walker, loop_number=qmc.nwarmups)

    for i in 1 : qmc.nsamples
        sweep!(system, qmc, walker, loop_number=half_bins)

        update!(ρ₊, walker, 1)
        update!(ρ₋, walker, 2)
        measure_SpinCorr(sampler, ρ₊, ρ₋)
    end

    # store the measurement
    jldopen("$(path)/$(filename)", "w") do file
        write(file, "cicj", sampler.cᵢ₊ᵣcᵢ)
        write(file, "SiSj", sampler.Sᵢ₊ᵣSᵢ)
    end
end
