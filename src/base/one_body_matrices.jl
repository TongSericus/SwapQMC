"""
    Generator for kinetic matrices of different models
"""

function hopping_matrix_Hubbard_1d(
    L::Int, t::Float64;
    isOBC::Bool = true
)
    T = zeros(L, L)

    isOBC ? begin
        hop_ind_left = [CartesianIndex(i+1, (i+1)%L+1) for i in 0 : L-2]
        hop_ind_right = [CartesianIndex((i+1)%L+1, i+1) for i in 0 : L-2]
        hop_amp = [-t for _ in 0 : L-2]
    end :
    # periodic boundary condition
    begin
        hop_ind_left = [CartesianIndex(i+1, (i+1)%L+1) for i in 0 : L-1]
        hop_ind_right = [CartesianIndex((i+1)%L+1, i+1) for i in 0 : L-1]
        hop_amp = [-t for _ in 0 : L-1]
    end

    @views T[hop_ind_left] = hop_amp
    @views T[hop_ind_right] = hop_amp
    
    return T
end

function hopping_matrix_ssh_1d(
    L::Int, t::Float64, δt::Float64; 
    isOBC::Bool = true
)
    T = zeros(L, L)

    isOBC ? begin
        hop_ind_left = [CartesianIndex(i+1, (i+1)%L+1) for i in 0 : L-2]
        hop_ind_right = [CartesianIndex((i+1)%L+1, i+1) for i in 0 : L-2]
        hop_amp = [-t-δt*(-1)^i for i in 0 : L-2]
    end :
    # periodic boundary condition
    begin
        hop_ind_left = [CartesianIndex(i+1, (i+1)%L+1) for i in 0 : L-1]
        hop_ind_right = [CartesianIndex((i+1)%L+1, i+1) for i in 0 : L-1]
        hop_amp = [-t-δt*(-1)^i for i in 0 : L-1]
    end

    @views T[hop_ind_left] = hop_amp
    @views T[hop_ind_right] = hop_amp
    
    return T
end

function hopping_matrix_Hubbard_2d(Lx::Int, Ly::Int64, t::Float64)
    L = Lx * Ly
    T = zeros(L, L)

    x = collect(0:L-1) .% Lx       # x positions for sites
    y = div.(collect(0:L-1), Lx)   # y positions for sites
    T_x = (x .+ 1) .% Lx .+ Lx * y      # translation along x-direction
    T_y = x .+ Lx * ((y .+ 1) .% Ly)    # translation along y-direction

    hop_ind_left = [CartesianIndex(i+1, T_x[i+1] + 1) for i in 0 : L-1]
    hop_ind_right = [CartesianIndex(T_x[i+1] + 1, i+1) for i in 0 : L-1]
    hop_ind_down = [CartesianIndex(i+1, T_y[i+1] + 1) for i in 0 : L-1]
    hop_ind_up = [CartesianIndex(T_y[i+1] + 1, i+1) for i in 0 : L-1]

    @views T[hop_ind_left] .= -t
    @views T[hop_ind_right] .= -t
    @views T[hop_ind_down] .= -t
    @views T[hop_ind_up] .= -t
    
    return T
end

function hopping_matrix_SSH_2d(
    L::Tuple{Int, Int}, t::Float64, δt::Tuple{Float64, Float64};
    isOBC::Tuple{Bool, Bool} = (true, true)
)
    Lx, Ly = L
    V = prod(L)
    T = zeros(V, V)

    x = collect(0:V-1) .% Lx       # x positions for sites
    y = div.(collect(0:V-1), Lx)   # y positions for sites

    # open boundary condition in x
    x_ind = isOBC[1] ? findall(x -> x!=0, mod.(collect(1:V), Lx)) : collect(1:V)
    # open boundary condition in y
    y_ind = isOBC[2] ? collect(1:V-Ly) : collect(1:V)
    
    T_x = (x .+ 1) .% Lx .+ Lx * y    # translation along x-direction
    T_y = x .+ Lx * ((y .+ 1) .% Ly)  # translation along y-direction

    hop_ind_left = [CartesianIndex(i, T_x[i] + 1) for i in x_ind]
    hop_ind_right = [CartesianIndex(T_x[i] + 1, i) for i in x_ind]
    hop_ind_down = [CartesianIndex(i, T_y[i] + 1) for i in y_ind]
    hop_ind_up = [CartesianIndex(T_y[i] + 1, i) for i in y_ind]

    hop_amp_x = [-t-δt[1]*(-1)^i for i in x_ind .- 1]
    hop_amp_y = [-t-δt[2]*(-1)^(div(i, Ly)) for i in y_ind .- 1]

    @views T[hop_ind_left] = hop_amp_x
    @views T[hop_ind_right] = hop_amp_x
    @views T[hop_ind_up] = hop_amp_y
    @views T[hop_ind_down] = hop_amp_y
    
    return T
end
