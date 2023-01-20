
struct EtgData{T, E}
    """
    Preallocated data for Entanglement Measures
    """
    # Entanglement Hamiltonians
    HA::LDR{T, E}
    HA′::LDR{T, E}

    # temporal data for I-Gₐ
    ImGA::LDR{T, E}
    ImGA′::LDR{T, E}

    # temporal matrix for Poisson binomial distribution
    P::Matrix{ComplexF64}
end

struct EtgMeasure
    # transition probability
    p::Float64
    # particle-number distribution
    Pn2::Matrix{Float64}
end

"""
    Regular Renyi-2 Entropy, Grover's Estimator
"""
function measure_Grover_estimator()

end

"""
    Accessible Renyi-2 Entropy
"""
function measure_Pn2()
    """
    P_{n, 2} = exp(-S_{2, n}) / exp(-S_{2})
    """
end