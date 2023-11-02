"""
    Entanglement Entropy Measurements
"""

function measure_expS2!(
    sampler::EtgSampler, replica::Replica; 
    direction::Int = 2, forwardMeasurement::Bool = false
)
    s = sampler.s_counter[]
    p = sampler.p

    direction == 1 && begin
        update!(replica)
        p[s] = min(1, exp(-2 * replica.logdetGA[]))

        forwardMeasurement && (sampler.s_counter[] += 1)

        return nothing
    end

    p[s] = min(1, exp(2 * replica.logdetGA[]))
    
    forwardMeasurement && (sampler.s_counter[] += 1)

    return nothing
end

function measure_Pn!(sampler::EtgSampler, walker::HubbardWalker; forwardMeasurement::Bool = false)
    s = sampler.s_counter[]
    Pn₊ = sampler.Pn₊
    Pn₋ = sampler.Pn₋

    G = walker.G
    wsA = sampler.wsA

    Pn_estimator(G[1], sampler.Aidx, wsA, tmpPn=sampler.tmpPn)
    @views copyto!(Pn₊[:, s], sampler.tmpPn[:, end])
    Pn_estimator(G[2], sampler.Aidx, wsA, tmpPn=sampler.tmpPn)
    @views copyto!(Pn₋[:, s], sampler.tmpPn[:, end])

    forwardMeasurement && (sampler.s_counter[] += 1)

    return nothing
end

function measure_Pn2!(
    sampler::EtgSampler, replica::Replica; 
    Np::Int = size(replica.GA⁻¹,2), forwardMeasurement::Bool = false
)
    s = sampler.s_counter[]
    Pn2₊ = sampler.Pn₊
    Pn2₋ = sampler.Pn₋

    Pn2_estimator(replica, tmpPn=sampler.tmpPn)
    @views copyto!(Pn2₊[:, s], sampler.tmpPn[:, Np])
    @views copyto!(Pn2₋[:, s], conj(sampler.tmpPn[:, Np]))

    forwardMeasurement && (sampler.s_counter[] += 1)

    return nothing
end

"""
    Grover_estimator(GA₁, ImGA₁, GA₂, ImGA₂, ws)

    Stable calculation of the Grover's estimator det[GA₁GA₂ + (I- GA₁)(I - GA₂)]
"""
function Grover_estimator(
    GA₁::LDR{T, E}, ImGA₁::LDR{T, E}, 
    GA₂::LDR{T, E}, ImGA₂::LDR{T, E}, 
    ws::LDRWorkspace{T, E}; 
    U::LDR{T, E} = similar(GA₁), V::LDR{T, E} = similar(ImGA₁)
) where {T, E}
    mul!(U, GA₁, GA₂, ws)
    mul!(V, ImGA₁, ImGA₂, ws)
    det_UpV(U, V, ws)
end

### Symmetry-resolving Calculations ###
"""
    poissbino(ϵ::Vector)

    A regularized version of the recursive calculation for the Poisson binomial
    distribution give the unnormalized spectrum ϵ
"""
function poissbino(
    ϵ::AbstractVector{T};
    Ns::Int64 = length(ϵ),
    ν1::AbstractVector{T} = ϵ ./ (1 .+ ϵ),
    ν2::AbstractVector{T} = 1 ./ (1 .+ ϵ),
    P::AbstractMatrix{Tp} = zeros(eltype(ϵ), Ns + 1, Ns)
) where {T<:Number, Tp<:Number}
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

@inline fermilevel(ϵ::AbstractVector, N::Int) = begin
    Ns = length(ϵ)
    return sqrt(abs(ϵ[Ns - N + 1] * ϵ[Ns - N]))
end

"""
    poissbino_recursion(ϵ::Vector)

    Poisson binomial recursion with rescaled spectrum, which improves the accuracy of
    computing tail distributions at the expense of increased computational cost
"""
function poissbino_recursion(
    ϵ::AbstractVector{T};
    Ns::Int64 = length(ϵ),
    logPn::AbstractVector{Tp} = zeros(eltype(ϵ), Ns+1),
    Pμ::AbstractMatrix{Tp} = zeros(eltype(ϵ), Ns+1, Ns),
    isRescale::Bool = true
) where {T, Tp}
    isRescale || return poissbino(ϵ, P=Pμ)

    # 0-particle sector (empty)
    logPn[1] = -sum(log.(1 .+ ϵ))
    # Ns-particle sector (fully filled)
    logPn[Ns+1] = sum(log.(ϵ)) + logPn[1]
    # rescale the spectrum to precisely compute each entry in Pn
    for N = 1 : Ns-1
        # rescale the spectrum
        μ = fermilevel(ϵ, N)
        ϵ′ = ϵ / μ
        # recursion with the new spectrum
        poissbino(ϵ′, P=Pμ)
        # obtain normalization with respect to rescaled and original spectrum
        logZμ = sum(log.(1 .+ ϵ′))
        logZ₀ = sum(log.(1 .+ ϵ))
        # compute log(Pₙ)
        logPn[N+1] = logZμ - logZ₀ + log(Pμ[N+1, Ns]) + N*log(μ)
    end

    return logPn
end

"""
    Pn2_estimator(...)

    Stable calculation of P_{n, 2} := exp(-S_{2, n}) / exp(-S_{2}) through recursion,
    via the eigvalues of the entanglement Hamiltonian Hₐ = Gₐ₁(I - Gₐ₁)⁻¹Gₐ₂(I - Gₐ₂)⁻¹
"""
function Pn2_estimator(
    replica::Replica; 
    L::Int = length(replica.Aidx), 
    tmpPn::AbstractMatrix{ComplexF64} = zeros(ComplexF64, L+1, L)
)
    Aidx = replica.Aidx
    G₁ = replica.G₀1
    G₂ = replica.G₀2
    ws = replica.ws

    U, d, V = compute_etgHam(G₁, G₂, Aidx, ws)
    lmul!(Diagonal(d), V)
    H = ws.M
    mul!(H, V, U)
    ϵ = eigvals(H)
    idx = findall(x -> abs(x)>1e-10, ϵ)
    ϵ = sort(ϵ[idx], by=abs)
    ϵ = sort(1 ./ ϵ, by=abs)

    # apply Poisson binomial iterator
    poissbino(ϵ, P=tmpPn)

    return tmpPn
end

"""
    Pn_estimator(G, Aidx, ws)

    Stable calculation of the particle number distribution P_{n} through recursion,
    via the eigvalues of the entanglement Hamiltonian Hₐ = (Gₐ⁻¹ - I)
"""
function Pn_estimator(
    G::AbstractMatrix{T}, Aidx::Vector{Int}, ws::LDRWorkspace{T,E};
    L::Int = length(Aidx),
    tmpPn::AbstractMatrix{ComplexF64} = zeros(ComplexF64, L+1, L)
) where {T,E}
    # compute the SVD of GA
    GA = ws.M
    @views copyto!(GA, G[Aidx, Aidx])
    GA_svd = svd!(GA, alg = LinearAlgebra.QRIteration())
    U, d, V = GA_svd

    # compute the eigenvalues of GA via its SVD
    Vt = ws.M
    adjoint!(Vt, V)
    dVt = ws.M′
    mul!(dVt, Diagonal(d), Vt)
    HA = ws.M
    mul!(HA, dVt, U)
    ϵ = eigvals(HA)
    idx = findall(x -> abs(1-x)>1e-10, ϵ)
    ϵ = (1 .- ϵ)[idx]

    # compute the eigenvalues of (GA⁻¹ - I)⁻¹
    ϵ = 1 ./ (1 ./ ϵ .- 1)
    sort!(ϵ, by=abs)

    # apply Poisson binomial iterator
    poissbino(ϵ, P=tmpPn)

    return tmpPn
end
