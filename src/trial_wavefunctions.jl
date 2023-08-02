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
    nᵢ₋ = [system.N[2] / system.V for _ in 1:system.V]
    # initialize Hamiltonian
    H₊ = copy(T)
    H₊[diagind(H₊)] = U * nᵢ₊
    H₋ = copy(T)
    H₋[diagind(H₋)] = U * nᵢ₋

    # perform self-consistent iterations on ⟨nᵢ↑⟩ and ⟨nᵢ↓⟩
    nᵢ₊′ = HF_wf_solver(H₊, system.N[1])
    nᵢ₋′ = HF_wf_solver(H₋, system.N[2])

    while norm(nᵢ₊′ - nᵢ₊) > ϵ && norm(nᵢ₋′ - nᵢ₋) > ϵ
        copyto!(nᵢ₊, nᵢ₊′)
        H₊[diagind(H₊)] = U * nᵢ₊
        copyto!(nᵢ₋, nᵢ₋′)
        H₋[diagind(H₋)] = U * nᵢ₋

        nᵢ₊′ = HF_wf_solver(H₊, system.N[1])
        nᵢ₋′ = HF_wf_solver(H₋, system.N[2])
    end

    H₊[diagind(H₊)] = U * nᵢ₊
    H₋[diagind(H₋)] = U * nᵢ₋

    ϕ₊ = HF_wf_solver(H₊, system.N[1], returnWF=true)
    ϕ₋ = HF_wf_solver(H₋, system.N[2], returnWF=true)

    return [ϕ₊, ϕ₋]
end
