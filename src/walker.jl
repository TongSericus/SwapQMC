"""
    Define random walkers
"""
### Struct to store a string of matrices or factorizations ###
Base.@kwdef struct Cluster{T}
    B::Vector{T}
end

Base.prod(C::Cluster{T}, a::Vector{Int}) where T = @views prod(C.B[a])

Cluster(Ns::Int, N::Int) = Cluster(B = [Matrix(1.0I, Ns, Ns) for _ in 1 : N])
Cluster(A::Factorization{T}, N::Int) where T = Cluster(B = [similar(A) for _ in 1 : N])

### Random walker definitions ###
abstract type GCWalker end

struct HubbardGCWalker{Ts<:Number, T<:Number, Fact<:Factorization{T}, E, C} <: GCWalker
    """
        GC walker for Hubbard-type model where a fast rank-1 update is available,
        could be regular, ionic, bilayer, etc.
    """
    α::Matrix{Float64}

    # Statistical weights of the walker, stored in the logarithmic form, while signs are the phases
    weight::Vector{Float64}
    sign::Vector{Ts}

    # Use reference to make chemical potential tunable on the fly
    expβμ::Base.RefValue{Float64}

    auxfield::Matrix{Int64}
    F::Vector{Fact}
    ws::LDRWorkspace{T, E}
    G::Vector{Matrix{T}}

    ### Temporal data to avoid memory allocations ###
    # All partial factorizations
    FC::Cluster{Fact}
    # Factorization of two unit matrices for spin-up and spin-down
    Fτ::Vector{Fact}
    # Temporal array of matrices with the ith element B̃_i being
    # B̃_i = B_{(cidx-1)k + i}, where cidx is the index of the current imaginary time slice to be updated
    # Note that the spin-up and spin-down matrices are strored as the first and the second half of the array, respectively
    Bl::Cluster{C}
    # Temporal array of matrices with the ith element B̃_i being
    # B̃_i = B_{(i-1)k + 1} ⋯ B_{ik}, where k is the stablization interval defined as qmc.stab_interval
    Bc::Cluster{C}
end

function HubbardGCWalker(
    system::System, qmc::QMC;
    auxfield::Matrix{Int} = 2 * (rand(system.V, system.L) .< 0.5) .- 1, 
    μ::Float64 = system.μ,
    T::DataType = Float64
)
    """
        Initialize a Hubbard-type GC walker
    """
    Ns = system.V
    k = qmc.stab_interval

    weight = zeros(Float64, 2)
    sign = zeros(T, 2)

    G = [Matrix{T}(1.0I, Ns, Ns), Matrix{T}(1.0I, Ns, Ns)]
    ws = ldr_workspace(G[1])
    F, Bc, FC = run_full_propagation_reverse(auxfield, system, qmc, ws)

    Fτ = ldrs(G[1], 2)
    Bl = Cluster(Ns, 2 * k)

    expβμ = exp(system.β * μ)
    weight[1], sign[1] = inv_IpμA!(G[1], F[1], expβμ, ws)
    weight[2], sign[2] = inv_IpμA!(G[2], F[2], expβμ, ws)

    α = system.auxfield[1, 1] / system.auxfield[2, 1]
    α = [α - 1 1/α - 1; 1/α - 1 α - 1]

    return HubbardGCWalker(α, -weight, sign, Ref(expβμ), auxfield, F, ws, G, FC, Fτ, Bl, Bc)
end

### Swap walker definitions ###

abstract type Swapper end

struct HubbardGCSwapper{Ts<:Number, T<:Number, E<:Number, Fact<:Factorization{T}} <: Swapper
    weight::Vector{Float64}
    sign::Vector{Ts}
    F::Vector{Fact}
    ws::LDRWorkspace{T, E}
    G::Vector{Matrix{T}}

    # preallocated temporal data
    B::Matrix{Float64}
    C::Vector{Fact}
    L::Fact
    R::Fact
end

function HubbardGCSwapper(
    extsys::ExtendedSystem, 
    walker₁::HubbardGCWalker, walker₂::HubbardGCWalker,
    T::DataType = eltype(walker₁.sign)
)
    V = extsys.Vext
    expβμ = walker₁.expβμ[]

    B = Matrix{T}(1.0I, V, V)
    G = [Matrix{T}(1.0I, V, V), Matrix{T}(1.0I, V, V)]
    
    F = ldrs(B, 2)
    ws = ldr_workspace(B)
    C = ldrs(B, 2)
    L = ldr(B)
    R = ldr(B)

    # expand F in the spin-up part and then merge
    expand!(F[1], walker₁.F[1], 1, expβμ = expβμ)
    expand!(L, walker₂.F[1], 2, expβμ = expβμ)
    copyto!(C[1], L)
    lmul!(L, F[1], ws)
    # expand F in the spin-down part and then merge
    expand!(F[2], walker₁.F[2], 1, expβμ = expβμ)
    expand!(L, walker₂.F[2], 2, expβμ = expβμ)
    copyto!(C[2], L)
    lmul!(L, F[2], ws)

    # compute Green's function
    weight = zeros(Float64, 2)
    sign = zeros(T, 2)
    weight[1], sign[1] = inv_IpA!(G[1], F[1], ws)
    weight[2], sign[2] = inv_IpA!(G[2], F[2], ws)

    return HubbardGCSwapper(-weight, sign, F, ws, G, B, C, L, R)
end
