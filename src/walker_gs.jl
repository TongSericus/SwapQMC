"""
    Walker definition and operations in the ground state
"""

abstract type GSWalker end

struct HubbardWalker{T<:Number, wf<:AbstractMatrix, Fact<:Factorization{T}, C} <: GSWalker
    """
        Hubbard-type walker where the two-body term is only on-site and a fast
        update scheme is available
    """
    α::Matrix{T}

    # trial wavefunction
    φ₀::Vector{wf}
    φ₀T::Vector{wf}

    auxfield::Matrix{Int64}
    # factorization for the left propagator matrix
    Fl::Vector{Fact}
    # factorization for the right propagator matrix
    Fr::Vector{Fact}

    ws::LDRWorkspace{T}
    # Green's function and the temporal data to compute G
    G::Vector{Matrix{T}}
    Ul::Vector{Matrix{T}}
    Ur::Vector{Matrix{T}}

    # imaginary-time-displaced Green's
    Gτ0::Vector{Matrix{T}}
    G0τ::Vector{Matrix{T}}

    ### Temporal data to avoid memory allocations ###
    # a transient factorization
    Fτ::Vector{Fact}
    # partial factorizations for the left and the right propagator matrices
    Fcl::Cluster{Fact}
    Fcr::Cluster{Fact}
    # Temporal array of matrices for cluster sweep
    Bl::Cluster{C}
    # Temporal matrix to store the product of a cluster of matrices
    Bc::Vector{C}

    ### Date for debugging ###
    tmp_r::Vector{T}
end

"""
    HubbardWalker(system, qmc, φ₀)

    Initialize a Hubbard-type ground state walker given the model parameters
"""
function HubbardWalker(
    system::Hubbard, qmc::QMC, φ₀::Vector{wf};
    auxfield::Matrix{Int} = rand([-1, 1], system.V, system.L),
    T::DataType = eltype(system.auxfield)
) where {wf<:AbstractMatrix}

    Ns = system.V
    Np = system.N
    @assert size(φ₀[1]) == (Ns, Np[1]) && size(φ₀[1]) == (Ns, Np[2]) "Invalid trial wavefunction!"
    φ₀T = @. Matrix(transpose(φ₀))

    # initialize equal-time and time-displaced Green's functions
    G   = [Matrix{T}(1.0I, Ns, Ns), Matrix{T}(1.0I, Ns, Ns)]
    Gτ0 = [Matrix{T}(1.0I, Ns, Ns), Matrix{T}(1.0I, Ns, Ns)]
    G0τ = [Matrix{T}(1.0I, Ns, Ns), Matrix{T}(1.0I, Ns, Ns)]

    # build the initial propator with random configurations
    ws = ldr_workspace(G[1])
    θ = div(system.L, 2)
    @views Fr, Bcr, Fcr = build_propagator(
                            auxfield[:, 1:θ], system, qmc, ws,
                            isReverse=false,
                            K=div(qmc.K,2), 
                            K_interval=qmc.K_interval[1:div(qmc.K,2)]
                        )
    @views Fl, Bcl, Fcl = build_propagator(
                            auxfield[:, θ+1:end], system, qmc, ws,
                            K=div(qmc.K,2),
                            K_interval=qmc.K_interval[div(qmc.K,2)+1:end]
                        )
    # compute Green's function based on the propagator
    Ul = [Matrix{T}(1.0I, Np[1], Ns), Matrix{T}(1.0I, Np[2], Ns)]
    Ur = [Matrix{T}(1.0I, Ns, Np[1]), Matrix{T}(1.0I, Ns, Np[2])]
    compute_G!(G[1], φ₀[1], φ₀T[1], Ul[1], Ur[1], Fl[1], Fr[1])
    compute_G!(G[2], φ₀[2], φ₀T[2], Ul[2], Ur[2], Fl[2], Fr[2])
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

    # initialize temporal data for storage
    Fτ = ldrs(G[1], 3)
    copyto!.(Fτ[2:3], Fr)
    Bl = Cluster(Ns, 2 * qmc.stab_interval, T = T)
    Bc = [copy(Bl.B[1]), copy(Bl.B[1])]
    tmp_r = Vector{T}()

    return HubbardWalker{T, eltype(φ₀), eltype(Fl), eltype(Bl.B)}(
        α, φ₀, φ₀T, 
        auxfield, Fl, Fr, ws, 
        G, Ul, Ur, Gτ0, G0τ, 
        Fτ, Fcl, Fcr, Bl, Bc,
        tmp_r
    )
end
