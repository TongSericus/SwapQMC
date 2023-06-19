"""
    Define random walkers
"""
### Struct to store a string of matrices or factorizations ###
Base.@kwdef struct Cluster{T}
    B::Vector{T}
end

Base.prod(C::Cluster{T}, a::Vector{Int}) where T = @views prod(C.B[a])

Cluster(Ns::Int, N::Int; T::DataType = Float64) = (T == Float64) ? 
                                                Cluster(B = [Matrix((1.0I)(Ns)) for _ in 1 : N]) : 
                                                Cluster(B = [Matrix(((1.0+0.0im)*I)(Ns)) for _ in 1 : N])
Cluster(A::Factorization{T}, N::Int) where T = Cluster(B = [similar(A) for _ in 1 : N])

Base.cat(C1::Cluster{T}, C2::Cluster{T}; isSpinful::Bool = true) where T = begin
    isSpinful || return Cluster(B = vcat(C1.B, C2.B))
    θ1 = div(length(C1.B), 2)
    θ2 = div(length(C2.B), 2)
    return @views Cluster(B = vcat(C1.B[1:θ1], C2.B[1:θ2], C1.B[θ1+1:end], C2.B[θ2+1:end]))
end

### Random walker definitions ###
abstract type GCWalker end

# GC walker for Hubbard-type model where a fast rank-1 update is available,
# could be regular, ionic, bilayer, etc.
struct HubbardGCWalker{T<:Number, Fact<:Factorization{T}, E, C} <: GCWalker
    α::Matrix{T}

    # Statistical weights of the walker, stored in the logarithmic form, while signs are the phases
    weight::Vector{Float64}
    sign::Vector{T}

    auxfield::Matrix{Int64}
    F::Vector{Fact}
    ws::LDRWorkspace{T, E}
    G::Vector{Matrix{T}}

    # imaginary-time-displaced Green's
    Gτ0::Vector{Matrix{T}}
    G0τ::Vector{Matrix{T}}

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

    ### Date for debugging ###
    tmp_r::Vector{T}
end

"""
    HubbardGCWalker(s::System, q::QMC)

    Initialize a Hubbard-type GC walker given the model parameters
"""
function HubbardGCWalker(
    system::Hubbard, qmc::QMC;
    auxfield::Matrix{Int} = rand([-1, 1], system.V, system.L),
    T::DataType = eltype(system.auxfield)
)
    Ns = system.V
    k = qmc.stab_interval

    weight = zeros(Float64, 2)
    sgn = zeros(T, 2)

    G   = [Matrix{T}(1.0I, Ns, Ns), Matrix{T}(1.0I, Ns, Ns)]
    Gτ0 = [Matrix{T}(1.0I, Ns, Ns), Matrix{T}(1.0I, Ns, Ns)]
    G0τ = [Matrix{T}(1.0I, Ns, Ns), Matrix{T}(1.0I, Ns, Ns)]

    ws = ldr_workspace(G[1])
    F, Bc, FC = build_propagator(auxfield, system, qmc, ws)

    Fτ = ldrs(G[1], 2)
    Bl = Cluster(Ns, 2 * k, T = T)

    weight[1], sgn[1] = inv_IpA!(G[1], F[1], ws)
    weight[2], sgn[2] = inv_IpA!(G[2], F[2], ws)

    # G(τ=0, 0) = G(0)
    copyto!.(Gτ0, G)
    # G(0, τ=0) = G(0) - I
    copyto!.(G0τ, G)
    G0τ[1][diagind(G0τ[1])] .-= 1
    G0τ[2][diagind(G0τ[2])] .-= 1

    if system.useChargeHST
        α = system.auxfield[1] / system.auxfield[2]
        α = [α - 1 1/α - 1; α - 1 1/α - 1]
    else
        α = system.auxfield[1] / system.auxfield[2]
        α = [α - 1 1/α - 1; 1/α - 1 α - 1]
    end

    tmp_r = Vector{T}()

    return HubbardGCWalker(α, -weight, sgn, auxfield, F, ws, G, Gτ0, G0τ, FC, Fτ, Bl, Bc, tmp_r)
end

"""
    update!(a::Walker)

    Update the Green's function and weight of the walker
"""
function update!(walker::HubbardGCWalker; identicalSpin::Bool = false)
    weight = walker.weight
    sgn = walker.sign
    G = walker.G
    F = walker.F

    weight[1], sgn[1] = inv_IpA!(G[1], F[1], walker.ws)
    identicalSpin && begin
        weight[2] = weight[1]
        sgn[2] = sgn[1]
        @. weight *= -1

        copyto!(G[2], G[1])

        return nothing
    end

    weight[2], sgn[2] = inv_IpA!(G[2], F[2], walker.ws)

    @. weight *= -1

    return nothing
end

### Swap walker definitions ###
abstract type Swapper end

struct HubbardGCSwapper{T<:Number, E<:Number, Fact<:Factorization{T}} <: Swapper
    weight::Vector{Float64}
    sign::Vector{T}
    F::Vector{Fact}
    ws::LDRWorkspace{T, E}
    G::Vector{Matrix{T}}

    # preallocated temporal data
    B::Matrix{T}
    Bk::Vector{Matrix{Float64}}
    Bk⁻¹::Vector{Matrix{Float64}}
    C::Vector{Fact}
    L::Fact
    R::Fact
end

function HubbardGCSwapper(
    extsys::ExtendedSystem, 
    walker₁::HubbardGCWalker, walker₂::HubbardGCWalker,
    T::DataType = eltype(walker₁.sign)
)   
    system = extsys.system
    V = extsys.Vext

    B = Matrix{T}(1.0I, V, V)
    G = [Matrix{T}(1.0I, V, V), Matrix{T}(1.0I, V, V)]
    
    # expand Bk and Bk⁻¹
    Tk = eltype(system.Bk)
    LA = extsys.LA
    LB = extsys.LB
    Bk = [Matrix{Tk}(1.0I, V, V), Matrix{Tk}(1.0I, V, V)]
    expand!(Bk[1], system.Bk, LA, LB, 1)
    expand!(Bk[2], system.Bk, LA, LB, 2)
    Bk⁻¹ = [Matrix{Tk}(1.0I, V, V), Matrix{Tk}(1.0I, V, V)]
    expand!(Bk⁻¹[1], system.Bk⁻¹, LA, LB, 1)
    expand!(Bk⁻¹[2], system.Bk⁻¹, LA, LB, 2)

    F = ldrs(B, 2)
    ws = ldr_workspace(B)
    C = ldrs(B, 2)
    L = ldr(B)
    R = ldr(B)

    # expand F in the spin-up part and then merge
    expand!(F[1], walker₁.F[1], 1)
    expand!(L, walker₂.F[1], 2)
    copyto!(C[1], L)
    lmul!(L, F[1], ws)
    # expand F in the spin-down part and then merge
    expand!(F[2], walker₁.F[2], 1)
    expand!(L, walker₂.F[2], 2)
    copyto!(C[2], L)
    lmul!(L, F[2], ws)

    # compute Green's function and weight
    weight = zeros(Float64, 2)
    sign = zeros(T, 2)
    weight[1], sign[1] = inv_IpA!(G[1], F[1], ws)
    weight[2], sign[2] = inv_IpA!(G[2], F[2], ws)
    @. weight *= -1

    return HubbardGCSwapper(weight, sign, F, ws, G, B, Bk, Bk⁻¹, C, L, R)
end

"""
    update!(s::Swapper)

    Update the Green's function and weight of the swapper
"""
function update!(swapper::HubbardGCSwapper; identicalSpin::Bool = false)
    weight = swapper.weight
    sign = swapper.sign
    G = swapper.G

    F = swapper.F

    weight[1], sign[1] = inv_IpA!(G[1], F[1], swapper.ws)
    identicalSpin && begin
        weight[2] = weight[1]
        sign[2] = sign[1]
        @. weight *= -1

        copyto!(G[2], G[1])

        return nothing
    end

    weight[2], sign[2] = inv_IpA!(G[2], F[2], swapper.ws)

    @. weight *= -1

    return nothing
end

"""
    fill_swapper!(s::Swapper, a::Walker, b::Walker)

    Fill a (potentially empty) swapper with two walkers
"""
function fill_swapper!(
    swapper::HubbardGCSwapper, 
    walker₁::HubbardGCWalker, walker₂::HubbardGCWalker;
    identicalSpin::Bool = false
)
    F = swapper.F
    C = swapper.C
    L = swapper.L
    ws = swapper.ws

    # expand F in the spin-up part and then merge
    expand!(F[1], walker₁.F[1], 1)
    expand!(L, walker₂.F[1], 2)
    copyto!(C[1], L)
    lmul!(L, F[1], ws)

    identicalSpin && begin
        update!(swapper, identicalSpin=true)

        return nothing
    end

    # expand F in the spin-down part and then merge
    expand!(F[2], walker₁.F[2], 1)
    expand!(L, walker₂.F[2], 2)
    copyto!(C[2], L)
    lmul!(L, F[2], ws)

    # compute Green's function and weight
    update!(swapper)

    return nothing
end
