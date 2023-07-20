"""
    Separate file for measuring correlation functions
"""

struct CorrFuncSampler
    # counters
    s_counter::Base.RefValue{Int}

    # all possible δr vectors
    δr::Vector{Tuple{Int, Int}}

    ipδr::Matrix{Int}

    # observables
    cᵢ₊ᵣcᵢ::Matrix{ComplexF64}
    Sᵢ₊ᵣSᵢ::Matrix{ComplexF64}
end

function periodic_mod(x::Int, y::Int)
    x >= 0 && return x % y
    return (y + x) % y
end

function CorrFuncSampler(system::System, qmc::QMC; nsamples::Int = qmc.nsamples)
    # generate all translational vectors
    δrx_max = div(system.Ns[1],2)
    δry_max = div(system.Ns[2],2)
    δr = [(δrx, δry) for δrx in 0:δrx_max for δry in 0:δry_max]
    popat!(δr, findall(x -> x==(0,0), δr)[1])

    Lx, Ly, _ = system.Ns
    L = Lx * Ly
    x = collect(0:L-1) .% Lx       # x positions for sites
    y = div.(collect(0:L-1), Lx)   # y positions for sites

    ipδr = zeros(Int, L, length(δr))
    for (n, i) in enumerate(δr)
        δrx, δry = i
        @views copyto!(ipδr[:, n], periodic_mod.(x .+ δrx, Lx) .+ Lx*(periodic_mod.(y .+ δry, Ly)) .+ 1)
    end

    cᵢ₊ᵣcᵢ = zeros(ComplexF64, length(δr), nsamples)
    Sᵢ₊ᵣSᵢ = zeros(ComplexF64, length(δr), nsamples)

    return CorrFuncSampler(Ref(1), δr, ipδr, cᵢ₊ᵣcᵢ, Sᵢ₊ᵣSᵢ)
end

"""
    measure_SpinCorr(system::System)

    Compute second-order spin-order in z-direction:
    ⟨Sᵢ₊ᵣSᵢ⟩ = N⁻¹∑ᵢ⟨(nᵢ₊ᵣ↑-nᵢ₊ᵣ↓)(nᵢ↑-nᵢ↓)⟩
    
    and first-order spin correlation
    N⁻¹∑ᵢ⟨cᵢ₊ᵣ↑cᵢ↑ + cᵢ₊ᵣ↓cᵢ↓⟩
"""
function measure_SpinCorr(
    sampler::CorrFuncSampler, ρ₊::DensityMatrix, ρ₋::DensityMatrix
)
    s = sampler.s_counter[]
    ρ₁₊ = ρ₊.ρ₁
    ρ₁₋ = ρ₋.ρ₁
    cᵢ₊ᵣcᵢ = sampler.cᵢ₊ᵣcᵢ
    Sᵢ₊ᵣSᵢ = sampler.Sᵢ₊ᵣSᵢ

    Ns⁻¹ = 1 / length(sampler.ipδr[:, 1])
    @inbounds for n in 1:length(sampler.δr)
        for (i,ipδr) in enumerate(@view sampler.ipδr[:, n])
                Sᵢ₊ᵣSᵢ[n, s] += ρ₂(ρ₊, ipδr, ipδr, i, i) + ρ₂(ρ₋, ipδr, ipδr, i, i) - 
                                ρ₁₊[ipδr, ipδr] * ρ₁₋[i, i] - ρ₁₊[i, i] * ρ₁₋[ipδr, ipδr]
                cᵢ₊ᵣcᵢ[n, s] += ρ₁₊[ipδr, i] + ρ₁₋[ipδr, i]
        end
        Sᵢ₊ᵣSᵢ[n, s] *= Ns⁻¹
        cᵢ₊ᵣcᵢ[n, s] *= Ns⁻¹
    end

    sampler.s_counter[] += 1

    return nothing
end
