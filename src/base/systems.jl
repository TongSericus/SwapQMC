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
    useComplexHST::Bool
    auxfield::Vector{T}
    V₊::Vector{T}
    V₋::Vector{T}

    # if use first-order Trotterization: exp(-ΔτK) * exp(-ΔτV)
    useFirstOrderTrotter::Bool

    # kinetic propagator
    Bk::Matrix{Float64}
    Bk⁻¹::Matrix{Float64}

    function BilayerHubbard(
        Ns::Tuple{Int64, Int64}, N::Tuple{Int64, Int64},
        t::Float64, t′::Float64, U::Float64,
        μ::Float64,
        β::Float64, L::Int64;
        useComplexHST::Bool = false,
        useFirstOrderTrotter::Bool = false
    )  
        Δτ = β / L

        T = one_body_matrix_bilayer_hubbard(Ns[1], Ns[2], t, t′)
        useFirstOrderTrotter ? Bk = exp(-T * Δτ) : Bk = exp(-T * Δτ / 2)
        Bk⁻¹ = inv(Bk)

        ### HS transform ###
        # complex HS transform is more stable for entanglement measures
        # see PRE 94, 063306 (2016) for explanations
        if useComplexHST
            γ = acosh(exp(-Δτ * U / 2) + 0im)
            # use symmetric Hubbard potential
            auxfield = [exp(γ), exp(-γ)]
            sys_type = ComplexF64
        else
            γ = atanh(sqrt(tanh(Δτ * U / 4)))
            auxfield = [exp(2 * γ - Δτ * U / 2), exp(-2 * γ - Δτ * U / 2)]
            sys_type = Float64
        end

        # add chemical potential
        @. auxfield *= exp.(μ * Δτ)

        V = prod(Ns)*2
        V₊ = zeros(sys_type, V)
        V₋ = zeros(sys_type, V)

        return new{sys_type}(
            Ns, V, 
            N, t, t′, U,
            μ, β, L,
            useComplexHST, auxfield, V₊, V₋,
            useFirstOrderTrotter,
            Bk, Bk⁻¹
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
    useComplexHST::Bool
    auxfield::Vector{T}
    V₊::Vector{T}
    V₋::Vector{T}

    # if use first-order Trotterization: exp(-ΔτK) * exp(-ΔτV)
    useFirstOrderTrotter::Bool

    # kinetic propagator
    Bk::Matrix{Float64}
    Bk⁻¹::Matrix{Float64}
    # staggered potential chain
    BΔ::Vector{Float64}

    function IonicHubbard(
        Ns::Tuple{Int64, Int64}, N::Tuple{Int64, Int64},
        t::Float64, U::Float64,
        δ::Float64, μ::Float64,
        β::Float64, L::Int64;
        useComplexHST::Bool = false,
        useFirstOrderTrotter::Bool = false
    )  
        Δτ = β / L

        if Ns[2] == 1 
            T, Δ = one_body_matrix_ionic_hubbard_1D(Ns[1], t, δ)
        else
            T, Δ = one_body_matrix_ionic_hubbard_2D(Ns[1], Ns[2], t, δ)
        end
        useFirstOrderTrotter ? Bk = exp(-T * Δτ) : Bk = exp(-T * Δτ / 2)
        Bk⁻¹ = inv(Bk)
        BΔ = exp.(-Δ * Δτ)

        ### HS transform ###
        # complex HS transform is more stable for entanglement measures
        # see PRE 94, 063306 (2016) for explanations
        if useComplexHST
            # HS field is coupled to charge, which preserves SU(2) symmetry
            γ = acosh(exp(-Δτ * U / 2) + 0im)
            # use symmetric Hubbard potential
            auxfield = [exp(γ), exp(-γ)]
            sys_type = ComplexF64
        else
            # HS field is coupled to spin
            γ = atanh(sqrt(tanh(Δτ * U / 4)))
            auxfield = [exp(2 * γ - Δτ * U / 2), exp(-2 * γ - Δτ * U / 2)]
            sys_type = Float64
        end

        # add chemical potential
        @. auxfield *= exp.(μ * Δτ)

        V = prod(Ns)
        V₊ = zeros(sys_type, V)
        V₋ = zeros(sys_type, V)

        return new{sys_type}(
            Ns, V, 
            N, t, U, δ,
            μ, β, L,
            useComplexHST, auxfield, V₊, V₋,
            useFirstOrderTrotter,
            Bk, Bk⁻¹, BΔ
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
end

"""
    rearrange!(U, V, Aidx, Bidx)

    Rearrange the elements in V by putting elements whose indices are in Aidx into the upper (left) block
    of (matrix) U. Note that U and V should have the same size.
"""
function rearrange!(U::AbstractMatrix, V::AbstractMatrix, Aidx, Bidx)
    LA = length(Aidx)

    @views copyto!(U[1:LA, 1:LA], V[Aidx, Aidx])
    @views copyto!(U[1:LA, LA+1:end], V[Aidx, Bidx])
    @views copyto!(U[LA+1:end, 1:LA], V[Bidx, Aidx])
    @views copyto!(U[LA+1:end, LA+1:end], V[Aidx, Aidx])
end

function rearrange!(U::AbstractVector, V::AbstractVector, Aidx, Bidx)
    LA = length(Aidx)

    @views copyto!(U[1:LA], V[Aidx])
    @views copyto!(U[LA+1:end], V[Bidx])
end

function ExtendedSystem(system::Hubbard, Aidx::Vector{Int64}) 

    LA = length(Aidx)
    Bidx = findall(x -> !(x in Aidx), 1:prod(system.V))
    LB = length(Bidx)
    Vext = LA + 2 * LB

    LB == 0 && @error "Subsystem size should be smaller than system size"

    sys_type = typeof(system)
    NsX, NsY = system.Ns
    Δτ = system.β / system.L
    if sys_type <: BilayerHubbard
        T = one_body_matrix_bilayer_hubbard(NsX, NsY, system.t, system.t′)
        T′ = similar(T)

        # rearrange matrices based on the partition
        rearrange!(T′, T, Aidx, Bidx)

        system.useFirstOrderTrotter ? Bk = exp(-T′ * Δτ) : Bk = exp(-T′ * Δτ / 2)
        copyto!(system.Bk, Bk)
        copyto!(system.Bk⁻¹, inv(Bk))

        return ExtendedSystem{sys_type}(
            system,
            Aidx, LA, Bidx, LB, Vext
        )
    elseif sys_type <: IonicHubbard
        if NsY == 1 
            T, Δ = one_body_matrix_ionic_hubbard_1D(NsX, system.t, system.Δ)
        else
            T, Δ = one_body_matrix_ionic_hubbard_2D(NsX, NsY, system.t, system.Δ)
        end
        T′, Δ′ = similar(T), similar(Δ)

        # rearrange matrices based on the partition
        rearrange!(T′, T, Aidx, Bidx)
        rearrange!(Δ′, Δ, Aidx, Bidx)

        system.useFirstOrderTrotter ? Bk = exp(-T′ * Δτ) : Bk = exp(-T′ * Δτ / 2)
        copyto!(system.Bk, Bk)
        copyto!(system.Bk⁻¹, inv(Bk))
        copyto!(system.BΔ, exp.(-Δ * Δτ))

        return ExtendedSystem{sys_type}(
            system,
            Aidx, LA, Bidx, LB, Vext
        )
    end
end
