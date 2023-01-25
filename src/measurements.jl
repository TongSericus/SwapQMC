"""
    Entanglement Entropy Measurements
"""

struct EtgData{T, E}
    """
        Preallocated data for Entanglement Measures
    """
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

function EtgData(extsys::ExtendedSystem; T::DataType = Float64)
    LA = extsys.LA

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
    walker₁::HubbardGCWalker, walker₂::HubbardGCWalker
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

    update!(walker₁)
    update!(walker₂)

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
    p = sum(walker1.weight) + sum(walker2.weight) - sum(swapper.weight)
    p_sign = prod(walker1.sign) * prod(walker2.sign) * prod(swapper.sign)
    etgm.p[] = min(1, p_sign * exp(p))

    return nothing
end

function measure_EE(
    walker₁::HubbardGCWalker, walker₂::HubbardGCWalker,
    swapper::HubbardGCSwapper
)
    """
        Measure the transition probability in Z² space
    """

    fill!(swapper, walker₁, walker₂)
    p = sum(swapper.weight) - sum(walker1.weight) - sum(walker2.weight)
    p_sign = prod(walker1.sign) * prod(walker2.sign) * prod(swapper.sign)

    return min(1, p_sign * exp(p))
end

function Grover_estimator(
    GA₁::LDR{T, E}, ImGA₁::LDR{T, E}, 
    GA₂::LDR{T, E}, ImGA₂::LDR{T, E}, 
    ws::LDRWorkspace{T, E}; 
    U::LDR{T, E} = similar(GA₁), V::LDR{T, E} = similar(ImGA₁)
) where {T, E}
    """
        Compute det[GA₁GA₂ + (I- GA₁)(I - GA₂)]
    """
    mul!(U, GA₁, GA₂, ws)
    mul!(V, ImGA₁, ImGA₂, ws)
    det_UpV(U, V, ws)
end

"""
    Accessible Renyi-2 Entropy
"""

function poissbino(
    ϵ::AbstractVector{T};
    Ns::Int64 = length(ϵ),
    ν1::AbstractVector{T} = ϵ ./ (1 .+ ϵ),
    ν2::AbstractVector{T} = 1 ./ (1 .+ ϵ),
    P::AbstractMatrix{Tp} = zeros(eltype(ϵ), Ns + 1, Ns)
) where {T<:Number, Tp<:Number}
    """
        A regularized version of the recursive calculation for the Poisson binomial
    distribution
        For details on how this is employed in the canonical ensemble calculations,
    see doi.org/10.1103/PhysRevResearch.2.043206

        # Argument
        ϵ -> eigenvalues
    """
    # Initialization
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
        compute P_{n, 2} := exp(-S_{2, n}) / exp(-S_{2})
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

    return nothing
end
