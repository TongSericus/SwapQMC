struct EtgSampler
    # partition and measure point (space and time)
    Aidx::Vector{Int}
    mp_x::Int
    mp_t::Int
    
    # counters
    s_counter::Base.RefValue{Int} # count the number of collected samples
    m_counter::Base.RefValue{Int} # count the measurement interval

    # observables
    p::Vector{Float64} # transition probability
    Pn::Matrix{ComplexF64}  # probability distribution

    # temporal data
    tmpPn::Matrix{ComplexF64}
end

function EtgSampler(extsys::ExtendedSystem, qmc::QMC)
    Aidx = extsys.Aidx
    x = Aidx[end]
    θ = div(extsys.system.L, 2)

    p = zeros(qmc.nsamples)

    LA = length(Aidx)
    Pn = zeros(ComplexF64, LA+1, qmc.nsamples)
    tmpPn = zeros(ComplexF64, LA + 1, LA)

    return EtgSampler(Aidx, x, θ, Ref(1), Ref(0), p, Pn, tmpPn)
end

function replica_measure!(sampler::EtgSampler, replica::Replica)
    s = sampler.s_counter[]
    p = sampler.p

    update!(replica)
    p[s] = min(1, exp(-2 * replica.logdetGA[]))

    sampler.s_counter[] += 1
    sampler.m_counter[] = 0

    return nothing
end

function measure!(sampler::EtgSampler, replica::Replica)
    s = sampler.s_counter[]
    p = sampler.p
    Pn2 = sampler.Pn
    tmpPn = sampler.tmpPn

    update!(replica)

    p[s] = min(1, exp(2 * replica.logdetGA[]))
    Pn2_tmp = Pn2_estimator(replica, Pn2=tmpPn)
    @views copyto!(Pn2[:, s], Pn2_tmp[:,end])
    
    sampler.s_counter[] += 1
    sampler.m_counter[] = 0

    return nothing
end

function measure!(sampler::EtgSampler, walker::HubbardSubsysWalker)
    s = sampler.s_counter[]
    Pn = sampler.Pn
    tmpPn = sampler.tmpPn

    G = walker.G[1]
    wsA = walker.wsA

    Pn_tmp = Pn_estimator(G, sampler.Aidx, wsA, Pn=tmpPn)
    @views copyto!(Pn[:, s], Pn_tmp[:,end])

    sampler.s_counter[] += 1
    sampler.m_counter[] = 0

    return nothing
end
