"""
    Entanglement Entropy Measurements
"""

### Preallocate data for Entanglement Measures ###
struct EtgData{T, E}
    # entanglement Hamiltonians
    HA₁::LDR{T, E}
    HA₂::LDR{T, E}

    # temporal data for Gₐ
    GA₁::LDR{T, E}
    GA₂::LDR{T, E}

    # temporal data for I-Gₐ
    ImGA₁::LDR{T, E}
    ImGA₂::LDR{T, E}

    # temporal matrix for Poisson binomial distribution
    P::Matrix{ComplexF64}

    ws::LDRWorkspace{T, E}
end

function EtgData(extsys::ExtendedSystem)
    LA = extsys.LA

    T = eltype(extsys.system.auxfield)
    HA₁, HA₂, GA₁, GA₂, ImGA₁, ImGA₂ = ldrs(zeros(T, LA, LA), 6)
    ws = ldr_workspace(HA₁)
    P = zeros(ComplexF64, LA + 1, LA)

    return EtgData{T, Float64}(HA₁, HA₂, GA₁, GA₂, ImGA₁, ImGA₂, P, ws)
end

struct EtgMeasurement{T}
    # transition probability
    p::Base.RefValue{Float64}
    # particle-number distribution (spin-resolved)
    Pn2₊::Vector{T}
    Pn2₋::Vector{T}

    function EtgMeasurement(extsys::ExtendedSystem; T::DataType = ComplexF64)
        LA = extsys.LA

        Pn2₊ = zeros(T, LA + 1)
        Pn2₋ = zeros(T, LA + 1)

        return new{T}(Ref(0.0), Pn2₊, Pn2₋)
    end
end

function measure_EE!(
    etgm::EtgMeasurement,
    etgdata::EtgData, extsys::ExtendedSystem, 
    walker₁::HubbardGCWalker, walker₂::HubbardGCWalker,
    swapper::HubbardGCSwapper
)
    """
        Measure the transition probability and particle number distribution in Z_{A, 2} space
    """
    LA = extsys.LA

    HA₁ = etgdata.HA₁
    HA₂ = etgdata.HA₂
    GA₁ = etgdata.GA₁
    GA₂ = etgdata.GA₂
    ImGA₁ = etgdata.ImGA₁
    ImGA₂ = etgdata.ImGA₂
    ws = etgdata.ws

    Pn2₊ = etgm.Pn2₊
    Pn2₋ = etgm.Pn2₋

    update!(walker₁, identicalSpin=extsys.system.useComplexHST)
    update!(walker₂, identicalSpin=extsys.system.useComplexHST)

    G₁ = walker₁.G
    G₂ = walker₂.G

    ### Spin-up part ###
    # compute the sub Green's function
    @views ldr!(GA₁, G₁[1][1:LA, 1:LA], ws)
    @views ldr!(GA₂, G₂[1][1:LA, 1:LA], ws)

    # compute I - Gₐ
    ImA!(ImGA₁, GA₁, ws)
    ImA!(ImGA₂, GA₂, ws)

    # then measure Pn2₊
    Pn2_estimator(GA₁, ImGA₁, GA₂, ImGA₂, ws, HA₁ = HA₁, HA₂ = HA₂, P = etgdata.P)
    @views copyto!(Pn2₊, etgdata.P[:, end])

    ### Spin-down part ###
    # compute the sub Green's function
    @views ldr!(GA₁, G₁[2][1:LA, 1:LA], ws)
    @views ldr!(GA₂, G₂[2][1:LA, 1:LA], ws)

    # compute I - Gₐ
    ImA!(ImGA₁, GA₁, ws)
    ImA!(ImGA₂, GA₂, ws)
    
    # then measure Pn2₋
    Pn2_estimator(GA₁, ImGA₁, GA₂, ImGA₂, ws, HA₁ = HA₁, HA₂ = HA₂, P = etgdata.P)
    @views copyto!(Pn2₋, etgdata.P[:, end])

    # compute the transition probability
    p = sum(walker₁.weight) + sum(walker₂.weight) - sum(swapper.weight)
    etgm.p[] = min(1, exp(p))

    return nothing
end

function measure_EE!(
    etgm::EtgMeasurement,
    etgdata::EtgData, extsys::ExtendedSystem, 
    replica::Replica{W, ComplexF64, Float64}
) where W
    LA = extsys.LA

    HA₁ = etgdata.HA₁
    HA₂ = etgdata.HA₂
    GA₁ = etgdata.GA₁
    GA₂ = etgdata.GA₂
    ImGA₁ = etgdata.ImGA₁
    ImGA₂ = etgdata.ImGA₂
    ws = etgdata.ws

    Pn2₊ = etgm.Pn2₊
    Pn2₋ = etgm.Pn2₋

    G₁ = replica.G₀1
    G₂ = replica.G₀2

    ### Spin-up part ###
    # compute the sub Green's function
    @views ldr!(GA₁, G₁[1:LA, 1:LA], ws)
    @views ldr!(GA₂, G₂[1:LA, 1:LA], ws)

    # compute I - Gₐ
    ImA!(ImGA₁, GA₁, ws)
    ImA!(ImGA₂, GA₂, ws)

    # then measure Pn2
    Pn2_estimator(GA₁, ImGA₁, GA₂, ImGA₂, ws, HA₁ = HA₁, HA₂ = HA₂, P = etgdata.P)
    # spin-up and spin-down part are exactly the same for SU(2) transform
    @views copyto!(Pn2₊, etgdata.P[:, end])
    @views copyto!(Pn2₋, etgdata.P[:, end])

    # record the transition probability
    etgm.p[] = min(1, exp(2*replica.logdetGA[]))

    return nothing
end

function measure_EE(
    walker₁::HubbardGCWalker, walker₂::HubbardGCWalker,
    swapper::HubbardGCSwapper;
    isIdenticalSpin::Bool = false
)
    """
        Measure the transition probability in Z² space
    """
    fill_swapper!(swapper, walker₁, walker₂, identicalSpin=isIdenticalSpin)
    p = sum(swapper.weight) - sum(walker₁.weight) - sum(walker₂.weight)

    return min(1, exp(p))
end

function measure_Pn!(
    etgm::EtgMeasurement,
    etgdata::EtgData, extsys::ExtendedSystem, 
    walker::W
) where W
    # set alias
    LA = extsys.LA
    GA = etgdata.GA₁
    ImGA = etgdata.ImGA₁
    ws = etgdata.ws
    Pn2₊ = etgm.Pn2₊
    Pn2₋ = etgm.Pn2₋
    G₊ = walker.G[1]
    G₋ = walker.G[2]

    ### Spin-up part ###
    @views ldr!(GA, G₊[1:LA, 1:LA], ws)
    ImA!(ImGA, GA, ws)
    Pn_estimator(GA, ImGA, ws, HA = etgdata.HA₁, P = etgdata.P)
    @views copyto!(Pn2₊, etgdata.P[:, end])

    extsys.system.useComplexHST && (@views copyto!(Pn2₋, etgdata.P[:, end]); return nothing)

    ### Spin-down part ###
    @views ldr!(GA, G₋[1:LA, 1:LA], ws)
    ImA!(ImGA, GA, ws)
    Pn_estimator(GA, ImGA, ws, HA = etgdata.HA₁, P = etgdata.P)
    @views copyto!(Pn2₋, etgdata.P[:, end])

    return nothing
end

function Grover_estimator(
    GA₁::LDR{T, E}, ImGA₁::LDR{T, E}, 
    GA₂::LDR{T, E}, ImGA₂::LDR{T, E}, 
    ws::LDRWorkspace{T, E}; 
    U::LDR{T, E} = similar(GA₁), V::LDR{T, E} = similar(ImGA₁)
) where {T, E}
    """
        Grover_estimator(GA₁, ImGA₁, GA₂, ImGA₂, ws)

        Stable calculation of the Grover's estimator det[GA₁GA₂ + (I- GA₁)(I - GA₂)]
    """
    mul!(U, GA₁, GA₂, ws)
    mul!(V, ImGA₁, ImGA₂, ws)
    det_UpV(U, V, ws)
end

### Symmetry-resolved Calculations ###
function poissbino(
    ϵ::AbstractVector{T};
    Ns::Int64 = length(ϵ),
    ν1::AbstractVector{T} = ϵ ./ (1 .+ ϵ),
    ν2::AbstractVector{T} = 1 ./ (1 .+ ϵ),
    P::AbstractMatrix{Tp} = zeros(eltype(ϵ), Ns + 1, Ns)
) where {T<:Number, Tp<:Number}
    """
        poissbino(ϵ::Vector)

        A regularized version of the recursive calculation for the Poisson binomial
    distribution give the unnormalized spectrum ϵ
    """
    # Initialization
    fill!(P, zero(eltype(P)))
    P[1, 1] = ν2[1]
    P[2, 1] = ν1[1]
    # iteration over trials
    @inbounds for i = 2 : Ns
        P[1, i] = ν2[i] * P[1, i - 1]
        # iteration over number of successes
        for j = 2 : i + 1
            P[j, i] = ν2[i] * P[j, i - 1] + ν1[i] * P[j - 1, i - 1]
        end
    end

    return P
end

function Pn2_estimator(
    GA₁::LDR{T, E}, ImGA₁::LDR{T, E}, 
    GA₂::LDR{T, E}, ImGA₂::LDR{T, E}, 
    ws::LDRWorkspace{T, E}; 
    HA₁::LDR{T, E} = similar(GA₁), HA₂::LDR{T, E} = similar(ImGA₁),
    LA = length(GA₁.d),
    P::AbstractMatrix{ComplexF64} = zeros(ComplexF64, LA + 1, LA)
) where {T, E}
    """
        Pn2_estimator(GA₁, ImGA₁, GA₂, ImGA₂, ws)

        Stable calculation of P_{n, 2} := exp(-S_{2, n}) / exp(-S_{2}) through recursion,
    via the eigvalues of the entanglement Hamiltonian Hₐ = Gₐ₁(I - Gₐ₁)⁻¹Gₐ₂(I - Gₐ₂)⁻¹
    """
    copyto!(HA₁, GA₁)
    copyto!(HA₂, GA₂)

    # compute Hₐ = Gₐ₁(I - Gₐ₁)⁻¹Gₐ₂(I - Gₐ₂)⁻¹
    rdiv!(HA₁, ImGA₁, ws)
    rdiv!(HA₂, ImGA₂, ws)
    HA = HA₁
    rmul!(HA, HA₂, ws)

    # diagonalize, then apply the Poisson binomial iterator
    ϵ = eigvals(HA)
    poissbino(ϵ, P=P)

    return P
end

function Pn_estimator(
    GA::LDR{T, E}, ImGA::LDR{T, E}, 
    ws::LDRWorkspace{T, E}; 
    HA::LDR{T, E} = similar(GA),
    LA = length(GA.d),
    P::AbstractMatrix{ComplexF64} = zeros(ComplexF64, LA + 1, LA)
) where {T, E}
    """
        Pn_estimator(GA, ImGA, ws)

        Stable calculation of the particle number distribution P_{n} through recursion,
    via the eigvalues of the entanglement Hamiltonian Hₐ = Gₐ(I - Gₐ)⁻¹
    """
    copyto!(HA, GA)

    # compute Hₐ = Gₐ(I - Gₐ)⁻¹
    rdiv!(HA, ImGA, ws)

    # diagonalize, then apply the Poisson binomial iterator
    ϵ = eigvals(HA)
    poissbino(ϵ, P=P)

    return P
end
