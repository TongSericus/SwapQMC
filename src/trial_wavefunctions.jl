"""
    Generator for trial wavefunctions
"""

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