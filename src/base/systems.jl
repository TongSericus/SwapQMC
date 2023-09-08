"""
    Define the regular systems
"""

abstract type System end
abstract type Hubbard <: System end

##### Define the Hubbard model in the generic form (constant U, user-defined hopping) #####
struct GenericHubbard{T, Tk} <: Hubbard
    ### Model Constants ###
    Ns::Tuple{Int64, Int64, Int64}  # 3D
    V::Int64    # total dimension
    N::Tuple{Int64, Int64}  # spin-up and spin-dn
    T::Tk   # kinetic matrix (can be various forms)
    U::Float64

    ### Temperature and Chemical Potential ###
    μ::Float64
    β::Float64
    L::Int64
    
    ### Automatically-generated Constants ###
    useChargeHST::Bool
    auxfield::Vector{T}
    V₊::Vector{T}
    V₋::Vector{T}

    # if use first-order Trotterization: exp(-ΔτK) * exp(-ΔτV)
    useFirstOrderTrotter::Bool

    # kinetic propagator
    Bk::Tk
    Bk⁻¹::Tk

    function GenericHubbard(
        Ns::Tuple{Int64, Int64, Int64}, N::Tuple{Int64, Int64},
        T::AbstractMatrix, U::Float64,
        μ::Float64, β::Float64, L::Int64;
        sys_type::DataType = ComplexF64,
        useChargeHST::Bool = false,
        useFirstOrderTrotter::Bool = false
    )
        Δτ = β / L
        useFirstOrderTrotter ? dτ = Δτ : dτ = Δτ/2
        Bk = exp(-T * dτ)
        Bk⁻¹ = exp(T * dτ)

        ### HS transform ###
        # complex HS transform is more stable for entanglement measures
        # see PRE 94, 063306 (2016) for explanations
        if useChargeHST
            γ = sys_type == ComplexF64 ? acosh(exp(-Δτ * U / 2) + 0im) : acosh(exp(-Δτ * U / 2))
            # use symmetric Hubbard potential
            auxfield = [exp(γ), exp(-γ)]
        else
            #γ = atanh(sqrt(tanh(Δτ * U / 4)))
            #auxfield = [exp(2 * γ - Δτ * U / 2), exp(-2 * γ - Δτ * U / 2)]
            γ = sys_type == ComplexF64 ? acosh(exp(Δτ * U / 2) + 0im) : acosh(exp(Δτ * U / 2))
            auxfield = [exp(γ), exp(-γ)]
        end

        # add chemical potential
        @. auxfield *= exp.(μ * Δτ)

        V = prod(Ns)
        V₊ = zeros(sys_type, V)
        V₋ = zeros(sys_type, V)

        return new{sys_type, typeof(Bk)}(
            Ns, V, 
            N, T, U,
            μ, β, L,
            useChargeHST, auxfield, V₊, V₋,
            useFirstOrderTrotter,
            Bk, Bk⁻¹
        )
    end
end

##### Define the extended systems for swap algorithm #####
struct ExtendedSystem{T}
    ### Original System ###
    system::T

    ### Entanglement-related parameters ###
    Aidx::Vector{Int64}         # entangled region
    LA::Int64                   # size of the entangled region
    Bidx::Vector{Int64}         # rest of the system
    LB::Int64                   # size of the rest of the system 
    Vext::Int64                 # size of the enlarged system
end

"""
    rearrange!(U, V, Aidx, Bidx)

    Rearrange the elements in V by putting elements whose indices are in Aidx into the upper left block
    of matrix U. Note that U and V should have the same size.
"""
function rearrange!(U::AbstractMatrix, V::AbstractMatrix, Aidx::Vector{Int64}, Bidx::Vector{Int64})
    LA = length(Aidx)

    @views copyto!(U[1:LA, 1:LA], V[Aidx, Aidx])
    @views copyto!(U[1:LA, LA+1:end], V[Aidx, Bidx])
    @views copyto!(U[LA+1:end, 1:LA], V[Bidx, Aidx])
    @views copyto!(U[LA+1:end, LA+1:end], V[Bidx, Bidx])
end

"""
    rearrange!(U, V, Aidx, Bidx)

    Rearrange the elements in V by putting elements whose indices are in Aidx into the upper block
    of vector U. Note that U and V should have the same length.
"""
function rearrange!(U::AbstractVector, V::AbstractVector, Aidx::Vector{Int64}, Bidx::Vector{Int64})
    LA = length(Aidx)

    @views copyto!(U[1:LA], V[Aidx])
    @views copyto!(U[LA+1:end], V[Bidx])
end

function ExtendedSystem(system::Hubbard, Aidx::Vector{Int64}; subsysOrdering::Bool = true) 

    LA = length(Aidx)
    Bidx = findall(x -> !(x in Aidx), 1:system.V)
    LB = length(Bidx)
    Vext = LA + 2 * LB

    LB == 0 && @error "Subsystem size should be smaller than system size"

    sys_type = typeof(system)
    subsysOrdering || return ExtendedSystem{sys_type}(system, Aidx, LA, Bidx, LB, Vext)

    Δτ = system.β / system.L
    T′ = similar(system.T)

    # rearrange matrices based on the partition
    rearrange!(T′, system.T, Aidx, Bidx)

    system.useFirstOrderTrotter ? Bk = exp(-T′ * Δτ) : Bk = exp(-T′ * Δτ / 2)
    copyto!(system.Bk, Bk)
    copyto!(system.Bk⁻¹, inv(Bk))
        
    return ExtendedSystem{sys_type}(
        system,
        Aidx, LA, Bidx, LB, Vext
    )
end
