abstract type TunableWalker <: GCWalker end

function unload_μ!(system::Hubbard)
    Δτ = system.β / system.L
    @. system.auxfield /= exp.(system.μ * Δτ)

    return nothing
end

struct TunableHubbardWalker{Ts<:Number, T<:Number, Fact<:Factorization{T}, E, C} <: GCWalker
    α::Matrix{Float64}

    weight::Vector{Float64}
    sign::Vector{Ts}
    
    # Use reference to make chemical potential tunable on the fly
    expβμ::Base.RefValue{Float64}

    auxfield::Matrix{Int64}
    F::Vector{Fact}
    ws::LDRWorkspace{T, E}
    G::Vector{Matrix{T}}

    FC::Cluster{Fact}
    Fτ::Vector{Fact}
    Bl::Cluster{C}
    Bc::Cluster{C}
end

"""
    TunableHubbardWalker(s::System, q::QMC)

    Initialize a Hubbard-type GC walker with tunable chemical potential
"""
function TunableHubbardWalker(
    system::Hubbard, qmc::QMC;
    auxfield::Matrix{Int} = 2 * (rand(system.V, system.L) .< 0.5) .- 1, 
    μ::Float64 = system.μ,
    T::DataType = Float64
)
    Ns = system.V
    k = qmc.stab_interval

    weight = zeros(Float64, 2)
    sgn = zeros(T, 2)

    G = [Matrix{T}(1.0I, Ns, Ns), Matrix{T}(1.0I, Ns, Ns)]
    ws = ldr_workspace(G[1])
    F, Bc, FC = build_propagator(auxfield, system, qmc, ws)

    Fτ = ldrs(G[1], 2)
    Bl = Cluster(Ns, 2 * k)

    expβμ = exp(system.β * μ)
    weight[1], sgn[1] = inv_IpμA!(G[1], F[1], expβμ, ws)
    weight[2], sgn[2] = inv_IpμA!(G[2], F[2], expβμ, ws)

    α = system.auxfield[1, 1] / system.auxfield[2, 1]
    α = [α - 1 1/α - 1; 1/α - 1 α - 1]

    return TunableHubbardWalker(α, -weight, sgn, Ref(expβμ), auxfield, F, ws, G, FC, Fτ, Bl, Bc)
end

"""
    update!(a::TunableWalker)

    Update the Green's function and weight of the walker
"""
function update!(walker::TunableHubbardWalker)
    """
        Update the Green's function, weight of the walker
    """
    weight = walker.weight
    sgn = walker.sign
    expβμ = walker.expβμ[]
    G = walker.G
    F = walker.F

    weight[1], sgn[1] = inv_IpμA!(G[1], F[1], expβμ, walker.ws)
    weight[2], sgn[2] = inv_IpμA!(G[2], F[2], expβμ, walker.ws)

    @. weight *= -1

    return nothing
end

### Green's function calculation with dynamical μ ###
function inv_IpμA!(G::AbstractMatrix{T}, A::LDR{T,E}, expβμ::Float64, ws::LDRWorkspace{T,E})::Tuple{E,T} where {T,E}

    Lₐ = A.L
    dₐ = expβμ * A.d
    Rₐ = A.R

    # calculate Rₐ⁻¹
    Rₐ⁻¹ = ws.M′
    copyto!(Rₐ⁻¹, Rₐ)
    logdetRₐ⁻¹, sgndetRₐ⁻¹ = Slinalg.inv_lu!(Rₐ⁻¹, ws.lu_ws)

    # calculate D₋ = min(Dₐ, 1)
    d₋ = ws.v
    @. d₋ = min(dₐ, 1)

    # calculate Lₐ⋅D₋
    Slinalg.mul_D!(ws.M, Lₐ, d₋)

    # calculate D₊ = max(Dₐ, 1)
    d₊ = ws.v
    @. d₊ = max(dₐ, 1)

    # calculate sign(det(D₊)) and log(|det(D₊)|)
    logdetD₊, sgndetD₊ = Slinalg.det_D(d₊)

    # calculate Rₐ⁻¹⋅D₊⁻¹
    Rₐ⁻¹D₊ = Rₐ⁻¹
    Slinalg.rdiv_D!(Rₐ⁻¹D₊, d₊)

    # calculate M = Rₐ⁻¹⋅D₊⁻¹ + Lₐ⋅D₋
    @. ws.M += Rₐ⁻¹D₊

    # calculate M⁻¹ = [Rₐ⁻¹⋅D₊⁻¹ + Lₐ⋅D₋]⁻¹
    M⁻¹ = ws.M
    logdetM⁻¹, sgndetM⁻¹ = Slinalg.inv_lu!(M⁻¹, ws.lu_ws)

    # calculate G = Rₐ⁻¹⋅D₊⁻¹⋅M⁻¹
    mul!(G, Rₐ⁻¹D₊, M⁻¹)

    # calculate sign(det(G)) and log(|det(G)|)
    sgndetG = sgndetRₐ⁻¹ * conj(sgndetD₊) * sgndetM⁻¹
    logdetG = logdetRₐ⁻¹ - logdetD₊  + logdetM⁻¹

    return real(logdetG), sgndetG
end
