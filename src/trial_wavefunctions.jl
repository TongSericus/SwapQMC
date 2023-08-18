"""
    Generator for trial wavefunctions
"""

### Free Trial Wave Function ###

"""
    trial_wf_free(system, T)
    Generate the trial wavefunction in the simplest form

    T -> hopping (kinetic) matrix
"""
function trial_wf_free(system::System, spin::Int, T::AbstractMatrix)
    wf_hopping = copy(T)
    # creat a small flux
    map!(x -> x *= (1+0.05*rand()), wf_hopping, T)

    # force Hermiticity
    wf_hopping += wf_hopping'
    wf_hopping /= 2
    wf_eig = eigen(wf_hopping)

    return wf_eig.vectors[:, 1:system.N[spin]]
end

### Hatree Fock Trial Wave Function ###

function HF_wf_solver(H::AbstractMatrix, N::Int; returnWF::Bool=false)
    H_eig = eigen(H)
    ϕ  = H_eig.vectors[:,1:N]
    returnWF && return ϕ
    ϕᵀ = Matrix(transpose(ϕ))
    return diag(ϕ * inv(ϕᵀ * ϕ) * ϕᵀ)
end

"""
    trial_wf_HF(system::System)

    Generate the trial wavefunction using Hartree-Fock form
"""
function trial_wf_HF(system::GenericHubbard; ϵ::Float64=1e-5)
    U = system.U
    T = system.T

    # initialize ⟨nᵢ↑⟩ and ⟨nᵢ↓⟩
    nᵢ₊ = [system.N[1] / system.V for _ in 1:system.V]
    # initialize Hamiltonian
    H₊ = copy(T)
    H₊[diagind(H₊)] = U * nᵢ₊

    # perform self-consistent iterations on ⟨nᵢ↑⟩ and ⟨nᵢ↓⟩
    nᵢ₊′ = HF_wf_solver(H₊, system.N[1])

    while norm(nᵢ₊′ - nᵢ₊) > ϵ
        copyto!(nᵢ₊, nᵢ₊′)
        H₊[diagind(H₊)] = U * nᵢ₊

        nᵢ₊′ = HF_wf_solver(H₊, system.N[1])
    end

    H₊[diagind(H₊)] = U * nᵢ₊

    ϕ₊ = HF_wf_solver(H₊, system.N[1], returnWF=true)
    ϕ₋ = copy(ϕ₊)

    return [ϕ₊, ϕ₋]
end

### BCS trial wavefunction ###
"""
    bcs_params(system::Hubbard)

    Self-consistently solve for the BCS superconducting gap, Δ, from the mean-filed theory,
    Currently only coded for 2D case but is easily generalizable to 3D
"""
function bcs_params(Lx::Int, Ly::Int, t::Float64, U::Float64)

    L = Lx*Ly

    # wave numbers for PBC
    kx, ky = collect(0.0 : 1/Lx : 1-1/Lx), collect(0.0 : 1/Ly : 1-1/Ly)
    # hopping energy as a function of wave numbers
    Ehop = [-2t*(cos(2π * kx[Nx]) + cos(2π * ky[Ny])) for Nx in 1:Lx for Ny in 1:Ly]

    # self-consistently solving for Δsc
    Δ = abs(U)
    Ek = sqrt.(Ehop.^2 .+ Δ^2)
    Δ′ = -U * sum(Δ ./ (2*Ek*L))
    while abs(Δ - Δ′) > 1e-10
        Δ = Δ′
        Ek = sqrt.(Ehop.^2 .+ Δ^2)
        Δ′ = -U * sum(Δ ./ (2*Ek*L))
    end

    return Δ
end

function trial_wf_bcs(system::GenericHubbard; t::Float64 = 1.0)
    
    # set alias
    Lx, Ly = system.Ns[1], system.Ns[2]
    L = system.V    # number of sites
    T = system.T
    U = system.U
    @assert U < 0

    Δsc = bcs_params(Lx, Ly, t, U)
    Δ = [Δsc for _ in 1:L]

    H = zeros(Float64, 2L, 2L)
    @views copyto!(H[1:L, 1:L], T)
    @views copyto!(H[L+1:2L, L+1:2L], -T)
    @views copyto!(H[1:L, L+1:2L], Diagonal(Δ))
    @views copyto!(H[L+1:2L, 1:L], Diagonal(Δ))

    eigH = eigen(H)
    E = eigH.values
    # check that energies come in plus-minus pairs
    @assert maximum(abs.(reverse(E[1:L], 1) + E[L+1:2L])) < 1e-12

    evecs = eigH.vectors
    # U and V are defined by the positive-energy eigenvectors
    U = evecs[1:L, L+1:2L]
    V = evecs[L+1:2L, L+1:2L]
    # check that (v, -u)^* are indeed the correct negative-energy eigenvectors
    for j in 1:L
        evec_tmp = vcat(conj(V[:, j]), -conj(U[:, j]))
        @assert (maximum(abs.(H * evec_tmp - E[L-j+1] * evec_tmp)) < 1e-12) "$(j) is bad"
    end

    return [conj(V[:, 1:system.N[1]]), -conj(U[:, 1:system.N[2]])]
end