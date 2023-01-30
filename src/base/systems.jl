"""
    Define the regular systems
"""

abstract type System end
abstract type Hubbard <: System end

struct BilayerHubbard{T} <: Hubbard
    ### Model Constants ###
    Ns::Tuple{Int64, Int64}
    V::Int64
    N::Tuple{Int64, Int64}
    t::Float64
    t′::Float64
    U::Float64

    ### Temperature and Chemical Potential ###
    μ::Float64
    β::Float64
    L::Int64
    
    ### Automatically-generated Constants ###
    auxfield::Matrix{Float64}
    V₊::Vector{T}
    V₋::Vector{T}

    # if use first-order Trotterization: exp(-ΔτK) * exp(-ΔτV)
    useFirstOrderTrotter::Bool

    # kinetic propagator
    Bk::Matrix{Float64}

    function BilayerHubbard(
        Ns::Tuple{Int64, Int64}, N::Tuple{Int64, Int64},
        t::Float64, t′::Float64, U::Float64,
        μ::Float64,
        β::Float64, L::Int64,
        useComplexHSTransform::Bool = false,
        useFirstOrderTrotter::Bool = true
    )  
        Δτ = β / L

        T = one_body_matrix_bilayer_hubbard(Ns[1], Ns[2], t, t′)
        useFirstOrderTrotter ? Bk = exp(-T * Δτ) : Bk = exp(-T * Δτ / 2)

        ### HS transform ###
        # complex HS transform is more stable for entanglement measures
        # see PRE 94, 063306 (2016) for explanations
        if useComplexHSTransform
            γ = acosh(complex(exp(-Δτ * U / 2)))
            auxfield = [
                exp(γ / 2 - Δτ * U / 4) exp(γ / 2 - Δτ * U / 4);
                exp(-γ / 2 - Δτ * U / 4) exp(-γ / 2 - Δτ * U / 4)
            ]
            sys_type = ComplexF64
        else
            γ = atanh(sqrt(tanh(Δτ * U / 4)))
            auxfield = [
                exp(2 * γ - Δτ * U / 2) exp(-2 * γ - Δτ * U / 2);
                exp(-2 * γ - Δτ * U / 2) exp(2 * γ - Δτ * U / 2)
            ]
            sys_type = Float64
        end

        V = prod(Ns)*2
        V₊ = zeros(sys_type, V)
        V₋ = zeros(sys_type, V)

        return new{sys_type}(
            Ns, V, 
            N, t, t′, U,
            μ, β, L,
            auxfield, V₊, V₋,
            useFirstOrderTrotter,
            Bk
        )
    end
end

struct IonicHubbard{T} <: Hubbard
    ### Model Constants ###
    Ns::Tuple{Int64, Int64}
    V::Int64
    N::Tuple{Int64, Int64}
    t::Float64
    U::Float64
    # staggered potential
    # one can restore the regular Hubbard model by setting this term to 0
    Δ::Float64

    ### Temperature and Chemical Potential ###
    μ::Float64
    β::Float64
    L::Int64
    
    ### Automatically-generated Constants ###
    auxfield::Vector{T}
    V₊::Vector{T}
    V₋::Vector{T}

    # if use first-order Trotterization: exp(-ΔτK) * exp(-ΔτV)
    useFirstOrderTrotter::Bool

    # kinetic propagator
    Bk::Matrix{Float64}
    # staggered potential chain
    BΔ::Vector{Float64}

    function IonicHubbard(
        Ns::Tuple{Int64, Int64}, N::Tuple{Int64, Int64},
        t::Float64, U::Float64,
        δ::Float64, μ::Float64,
        β::Float64, L::Int64,
        useComplexHSTransform::Bool = false,
        useFirstOrderTrotter::Bool = true
    )  
        Δτ = β / L

        T, Δ = one_body_matrix_ionic_hubbard_2D(Ns[1], Ns[2], t, δ)
        useFirstOrderTrotter ? Bk = exp(-T * Δτ) : Bk = exp(-T * Δτ / 2)
        BΔ = exp.(-Δ * Δτ)

        ### HS transform ###
        # complex HS transform is more stable for entanglement measures
        # see PRE 94, 063306 (2016) for explanations
        if useComplexHSTransform
            γ = acosh(complex(exp(-Δτ * U / 2)))
            auxfield = [exp(γ / 2 - Δτ * U / 4), exp(-γ / 2 - Δτ * U / 4)]
            sys_type = ComplexF64
        else
            γ = atanh(sqrt(tanh(Δτ * U / 4)))
            auxfield = [exp(2 * γ - Δτ * U / 2), exp(-2 * γ - Δτ * U / 2)]
            sys_type = Float64
        end

        V = prod(Ns)
        V₊ = zeros(sys_type, V)
        V₋ = zeros(sys_type, V)

        return new{sys_type}(
            Ns, V, 
            N, t, U, δ,
            μ, β, L,
            auxfield, V₊, V₋,
            useFirstOrderTrotter,
            Bk, BΔ
        )
    end
end

"""
    Define the extended systems for swap algorithm
"""

struct ExtendedSystem{T}
    ### Original System ###
    system::T

    ### Entanglement-related parameters ###
    Aidx::Vector{Int64}         # entangled region
    LA::Int64                   # size of the entangled region
    Bidx::Vector{Int64}         # rest of the system
    LB::Int64                   # size of the rest of the system 
    Vext::Int64                 # size of the enlarged system

    function ExtendedSystem(sys::System, Aidx::Vector{Int64}) 

        LA = length(Aidx)
        Bidx = findall(x -> !(x in Aidx), 1:prod(sys.V))
        LB = length(Bidx)
        Vext = LA + 2 * LB

        return new{typeof(sys)}(
            sys,
            Aidx, LA, Bidx, LB, Vext
        )
    end
end
