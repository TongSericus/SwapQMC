"""
    Define the regular systems
"""

abstract type System end

struct Hubbard <: System
end

struct IonicHubbard <: System
end

"""
    Define the extended systems for using swap algorithm
"""

abstract type ExtendedSystem end

struct HubbardExtended <: ExtendedSystem
end

struct IonicHubbardExtended <: ExtendedSystem
    ### Model Constants ###
    Ns::Tuple{Int64, Int64}
    V::Int64
    N::Tuple{Int64, Int64}
    t::Float64
    U::Float64
    Δ::Float64

    ### Temperature and Chemical Potential ###
    μ::Float64
    β::Float64
    L::Int64

    ### Entanglement-related parameters ###
    Aidx::Vector{Int64}         # entangled region
    LA::Int64                   # size of the entangled region
    Bidx::Vector{Int64}         # rest of the system
    LB::Int64                   # size of the rest of the system 
    Vext::Int64                 # size of the enlarged system
    
    ### Automatically-generated Constants ###
    auxfield::Matrix{Float64}

    # if use first-order Trotterization: exp(-ΔτK) * exp(-ΔτV)
    useFirstOrderTrotter::Bool

    # kinetic propagator
    Bk::Matrix{Float64}
    # staggered potential chain
    BΔ::Vector{Float64}

    function IonicHubbardExtended(
        Ns::Tuple{Int64, Int64}, N::Tuple{Int64, Int64},
        t::Float64, U::Float64,
        μ::Float64, δ::Float64,
        β::Float64, L::Int64,
        Aidx::Vector{Int64};
        useComplexHSTransform::Bool = false,
        useFirstOrderTrotter::Bool = true
    )  
        Δτ = β / L

        LA = length(Aidx)
        Bidx = findall(x -> !(x in Aidx), collect(1:prod(Ns)))
        LB = length(Bidx)
        Vext = LA + 2 * LB

        T, Δ = one_body_matrix_ionic_hubbard_2D(Ns[1], Ns[2], t, δ)
        useFirstOrderTrotter ? Bk = exp(-T * Δτ) : Bk = exp(-T * Δτ / 2)
        BΔ = exp.(-Δ * Δτ)

        ### HS transform ###
        # complex HS transform is more stable for entanglement measures
        # see PRE 94, 063306 (2016) for explanations
        if useComplexHSTransform
            γ = acosh(complex(exp(-Δτ * U / 2)))
            auxfield = [
                exp(γ / 2 - Δτ * U / 4) exp(γ / 2 - Δτ * U / 4);
                exp(-γ / 2 - Δτ * U / 4) exp(-γ / 2 - Δτ * U / 4)
            ]
        else
            γ = atanh(sqrt(tanh(Δτ * U / 4)))
            auxfield = [
                exp(2 * γ - Δτ * U / 2) exp(-2 * γ - Δτ * U / 2);
                exp(-2 * γ - Δτ * U / 2) exp(2 * γ - Δτ * U / 2)
            ]
        end

        return new(
            Ns, prod(Ns), 
            N, t, U, δ,
            μ, β, L, 
            Aidx, LA, Bidx, LB, Vext,
            auxfield,
            useFirstOrderTrotter,
            Bk, BΔ
        )
    end
end
