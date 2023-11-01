"""
    Replica{W, T, E}

Defines a replica with the given parameters.
"""
struct Replica{W, T, E}
    ### Partition indices ###
    Aidx::Vector{Int64}

    ### Two replica walkers ###
    walker1::W
    walker2::W

    ### Allocating Temporal Data ###
    # Green's functions at imaginary time θ/2
    G₀1::Matrix{T}
    G₀2::Matrix{T}
    
    # Inverse of the Grover matrix
    GA⁻¹::Matrix{T}
    logdetGA::Base.RefValue{Float64}    # note: it's the negative value log(detGA⁻¹)
    sgnlogdetGA::Base.RefValue{T}

    # Matrix I - 2*GA, where GA is the submatrix of either G₀1 or G₀2, depending on which replica is currently NOT being updated
    Im2GA::Matrix{T}

    # Allocating three vectors used in computing the ratio and updating the Grover inverse
    a::Vector{T}
    b::Vector{T}
    t::Vector{T}

    ### Thermaldynamic integration variable for incremental algorithm ###
    λₖ::Float64

    ### LDR Workspace ###
    ws::LDRWorkspace{T, E}

    function Replica(extsys::ExtendedSystem, w1::W, w2::W; λₖ::Float64 = 1.0) where W
        T = eltype(w1.G[1])

        LA = extsys.LA
        Aidx = extsys.Aidx
        GA⁻¹ = zeros(T, LA, LA)
        ws = ldr_workspace(GA⁻¹)
        G₀1, G₀2 = copy(w1.G[1]), copy(w2.G[1])
        logdetGA, sgnlogdetGA = inv_Grover!(GA⁻¹, G₀1[Aidx, Aidx], G₀2[Aidx, Aidx], ws)
        a, b, t = zeros(T, LA), zeros(T, LA), zeros(T, LA)
        Im2GA = I - 2 * G₀2[1:LA, 1:LA]

        return new{W, T, Float64}(
            Aidx, 
            w1, w2, 
            G₀1, G₀2, 
            GA⁻¹, Ref(logdetGA), Ref(sgnlogdetGA), 
            Im2GA, a, b, t, 
            λₖ, ws
        )
    end
end

### Display Info ###
Base.summary(r::Replica) = string(nameof(typeof(r)))

function Base.show(io::IO, r::Replica)
    println(io, TYPE_COLOR, Base.summary(r), NO_COLOR)
    println(io, "Partition size: ", TYPE_COLOR, length(r.Aidx), NO_COLOR)
    println(io, "log(detGA⁻¹): ", TYPE_COLOR, r.logdetGA[], NO_COLOR)
end

"""
    update!(r::Replica)

Updates the value of det(GA⁻¹) in the provided replica.
"""
function update!(r::Replica)
    Aidx, G₀1, G₀2, GA⁻¹, ws = r.Aidx, r.walker1.G[1], r.walker2.G[1], r.GA⁻¹, r.ws
    logdetGA, sgnlogdetGA = inv_Grover!(GA⁻¹, G₀1[Aidx, Aidx], G₀2[Aidx, Aidx], ws)
    r.logdetGA[] = logdetGA
    r.sgnlogdetGA[] = sgnlogdetGA
    copyto!(r.G₀1, G₀1)
    copyto!(r.G₀2, G₀2)

    return r
end
