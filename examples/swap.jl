using SwapQMC, JLD, InteractiveUtils

function run_swap_gce(
    extsys::ExtendedSystem{BilayerHubbard}, qmc::QMC, 
    direction::Int64, file_id::Int64
)
    """
        MC simulation for entanglement measurements
    """
    system = extsys.system

    ### Sweep in the Z_{A, 2} space ###
    if direction == 1

        etgdata = EtgData(extsys)
        etgm = EtgMeasurement(extsys)

        walker1 = HubbardGCWalker(system, qmc)
        walker2 = HubbardGCWalker(system, qmc)
        swapper = HubbardGCSwapper(extsys, walker1, walker2)

        p = zeros(Float64, qmc.nsamples)
        sgn = zeros(Float64, qmc.nsamples)
        Pn2_up = Matrix{Float64}(1.0I, extsys.LA+1, qmc.nsamples)
        Pn2_dn = Matrix{Float64}(1.0I, extsys.LA+1, qmc.nsamples)

        for i in 1 : qmc.nwarmups
            # sweep from 0 to β
            sweep!(extsys, qmc, swapper, walker1, 1)
            # sweep from β to 2β
            sweep!(extsys, qmc, swapper, walker2, 2)
        end

        for i in 1 : qmc.nsamples

            for j in 1 : qmc.measure_interval
                # sweep from 0 to β
                sweep!(extsys, qmc, swapper, walker1, 1)
                # sweep from β to 2β
                sweep!(extsys, qmc, swapper, walker2, 2)
            end
            
            # measurement
            measure_EE!(etgm, etgdata, extsys, walker1, walker2, swapper)

            # store the measurements
            p[i] = etgm.p[]
            sgn[i] = prod(swapper.sign)
            @views copyto!(Pn2_up[:, i], real(etgm.Pn2₊))
            @views copyto!(Pn2_dn[:, i], real(etgm.Pn2₋))
        end

        a = varinfo()
        b = varinfo(SwapQMC)

        #filename = "denom_tz$(system.t′)_U$(system.U)_beta$(system.β)_$(file_id).jld"
        #jldopen("../data/GCE_BiHub_Lx$(system.Ns[1])Ly$(system.Ns[2])_LA$(extsys.LA)/$filename", "w") do file
        #    write(file, "sgn", sgn)
        #    write(file, "p", p)
        #    write(file, "Pn2_up", Pn2_up)
        #    write(file, "Pn2_dn", Pn2_dn)
        #end

        filename = "denom_tz$(system.t′)_U$(system.U)_beta$(system.β)_$(file_id).jld"
        jldopen("../data/GCE_BiHub_Lx$(system.Ns[1])Ly$(system.Ns[2])_LA$(extsys.LA)/$filename", "w") do file
            write(file, "sgn", sgn)
            write(file, "p", p)
            write(file, "Pn2_up", Pn2_up)
            write(file, "Pn2_dn", Pn2_dn)
        end

    ### Sweep in the Z² space ###
    elseif direction == 2

        walker1 = HubbardGCWalker(system, qmc)
        walker2 = HubbardGCWalker(system, qmc)
        swapper = HubbardGCSwapper(extsys, walker1, walker2)

        p = zeros(Float64, qmc.nsamples)
        sgn = zeros(Float64, qmc.nsamples)

        for i in 1 : qmc.nwarmups
            sweep!(system, qmc, walker1)
            sweep!(system, qmc, walker2)
        end

        for i in 1 : qmc.nsamples

            for j in 1 : qmc.measure_interval
                sweep!(system, qmc, walker1)
                sweep!(system, qmc, walker2)
            end
            
            p[i] = measure_EE(walker1, walker2, swapper)
            sgn[i] = prod(walker1.sign) * prod(walker2.sign)
        end

        filename = "num_tz$(system.t′)_U$(system.U)_beta$(system.β)_$(file_id).jld"
        jldopen("./data/GCE_BiHub_Lx$(system.Ns[1])Ly$(system.Ns[2])_LA$(extsys.LA)/$filename", "w") do file
            write(file, "sgn", sgn)
            write(file, "p", p)
        end
    end
end
