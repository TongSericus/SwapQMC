"""
    Define the replica
"""

struct Replica{W<:GCWalker, T, E}
    # partition
    Aidx::Vector{Int64}

    # two replica walkers
    walker1::W
    walker2::W

    ### Temporal Data ###
    # Green's functions at imaginary time 0
    G₀1::Matrix{T}
    G₀2::Matrix{T}

    # inverse of the Grover matrix
    GA⁻¹::Matrix{T}
    logdetGA::Base.RefValue{Float64}
    sgnlogdetGA::Base.RefValue{T}

    # constant matrix I - 2*GA
    Im2GA::Matrix{T}

    # two vectors used in computing the ratio and updating the Grover inverse
    a::Vector{T}
    b::Vector{T}
    t::Vector{T}    # a temporal vector for storage

    # LDR Workspace
    ws::LDRWorkspace{T, E}

    function Replica(extsys::ExtendedSystem, walker1::W, walker2::W) where W

        T = eltype(walker1.G[1])
        LA = extsys.LA
        Aidx = collect(1:LA)

        GA⁻¹ = zeros(T, LA, LA)
        ws = ldr_workspace(GA⁻¹)
        G₀1 = copy(walker1.G[1])
        G₀2 = copy(walker2.G[1])
        @views logdetGA, sgnlogdetGA = inv_Grover!(GA⁻¹, G₀1[Aidx, Aidx], G₀2[Aidx, Aidx], ws)

        a = zeros(T, LA)
        b = zeros(T, LA)
        t = zeros(T, LA)

        Im2GA = I - 2 * (@view G₀2[1:LA, 1:LA])
        
        return new{W, T, Float64}(
            Aidx, 
            walker1, walker2, 
            G₀1, G₀2, 
            GA⁻¹, Ref(logdetGA), Ref(sgnlogdetGA), 
            Im2GA, 
            a, b, t, ws
        )
    end
end