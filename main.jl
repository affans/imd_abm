# Code to run, plot, and analyze large scale simulations (using ABM cluster)
# This file spawn workers and include the model on all workers when include(main.jl)

# these packages may not exist in the model environment files 
# make sure they are in the global environment to run the code
using Gnuplot # maybe in master environment
using ClusterManagers 
using Distributed
using DelimitedFiles

ENV["JULIA_WORKER_TIMEOUT"] = 120  ## julia 1.8 is taking longer to connect to workers

cluster_env = true
# setup parallel workers
if nprocs() == 1
    if gethostname() == "hpc" 
        @info "connecting to hpc"
        if cluster_env
            addprocs(SlurmManager(450), N=15, verbose="", topology=:master_worker, exeflags="--project=$(Base.active_project())")
        else 
            addprocs(10)
        end
    else 
        @info "connecting to local"
        addprocs(2, exeflags="--project=.") # for local usage
    end
    # if running Revise, all worker processes will be updated with new code
    @everywhere using Revise
    @everywhere includet("model.jl")
else 
    @info "processors already added"
end


# Launching sims 
julia> process_sims(xy, "adjusted")

#julia> xx = launch_sims()
#julia> process_sims(xx, "basecase") # will save files in the output folder
# 
# change p.adj_vax_cov for scenario analysis 
#julia> xy = launch_sims()
#julia> process_sims(xy, "adjusted")

function launch_sims(base=0.001, adj1=0.49, adj2=0.40, adj3=0.54)
    launch_sims(400, 500, base, adj1, adj2, adj3)
end

function launch_sims(nsims, maxtime, base, adj1, adj2, adj3)
    @info "Current process id: $(myid())"
    @info "Total number of processors: $(nprocs())"
    @info "Number of simulations: $nsims"

    # create an object for model parameters
    mp = ModelParameters()
    mp.sstime = maxtime
    mp.beta = [base*adj1, base*adj2, base*adj3] # set the beta values for the disease
    mp.adj_vax_cov = true # false for basecase
    println("Model parameters: $(mp)")

    # transfer/initialze the parameters over all the workers 
    # each process has its own `p` modelparameters
    @everywhere init_state($mp) # initialize state/parameters over all processors

    # to run steady state for long time
    # cd = pmap(1:nsims) do x
    #     @info "Starting simulation $x on host $(gethostname()), id: $(myid())"
    #     init_agents(x, false) # true for reading calibrated data, false for new population
    #     init_disease() # since we are reading calibrated data 
    #     steadystate(x, true) # true argument to save the calibrated files         
    # end

    cd = pmap(1:nsims) do x
        @info "Starting simulation $x on host $(gethostname()), id: $(myid())"
        init_agents(x, true) # true for reading calibrated data, false for new population
        vaccine_loop(x)
    end
    @everywhere flush(stdout)
    println("simulations finished - returning data")
    return cd
end

function process_sims(res, fprefix="") 
    # the resulting object has dimensions 
    # sim x time x agegroup x carriage x vaccine 
    # we want to split carriage/vaccine into it's own files 
    # and each file will have (time _ sim) x agegroup (sims are stacked on top of each other)

    # sim ident 
    simident = "r1"

    # get length/size inforamation
    nsim = length(res) 
    ntimeyrs, _, _, _ = size(res[1][2])  # since res[1][2] is the incidence matrix of the 1st simulation

    ### PROCESSING PREVALANCE
    # Let's save prevalance 
    # each prevalence object (say res[i][1] where i is a simulation) 
    # is a inftype x time (in weeks) object 
    # so res[i][1] each row represents sus, car1, car2, car3, imd, rec 
    # we want to take means/averages over simulations
    
    # for each simulation object s, s[1] gets the prevalence matrix 
    # s[1][2, :] extracts the car1 row (where s loops over the simulations)
    carc = reduce(hcat, [s[1][2, :] for s in res]) # creates a matrix of just C 
    carw = reduce(hcat, [s[1][3, :] for s in res]) # creates a matrix of just C 
    cary = reduce(hcat, [s[1][4, :] for s in res]) # creates a matrix of just C 
     
    # take average of each variable over all simulations, these will be a vector of length maxtime
    carc_avg = mean(carc, dims=2)
    carw_avg = mean(carw, dims=2)
    cary_avg = mean(cary, dims=2)
    
    # take the quants 
    carc_quant = mapslices(x -> quantile(x, [0.05, 0.95]), carc; dims=2)
    carw_quant = mapslices(x -> quantile(x, [0.05, 0.95]), carw; dims=2)
    cary_quant = mapslices(x -> quantile(x, [0.05, 0.95]), cary; dims=2)

    # repackage as a matrix to save as csv 
    scr = hcat(carc_avg, carc_quant, carw_avg, carw_quant, cary_avg, cary_quant)
    writedlm("./output/$(simident)_sim_avgprev_$(fprefix).csv", scr, ',')
    
    ### PROCESSING INCIDENCE
    # each incidence object (say res[i][2] where i is a simulation) 
    # is a tensor (4 dimensional): [time (in years), ag, carriage, vaccine]
    # so res[i][2][:, :, 1, 1] gives a matrix of (time x ag) for carriage1, with vaccine
    # so res[i][2][:, :, 1, 2] gives a matrix of (time x ag) for carriage1, no vaccine
    # so res[i][2][:, :, 2, 1] gives a matrix of (time x ag) for carriage2, with vaccine, 
    # so res[i][2][:, :, 2, 2] gives a matrix of (time x ag) for carriage2, with vaccine, 
    # For each simulation, we stack these matrices on top so 
    # if there are 10 years, 5 simulations, then there will be 50 rows 
    # each group of 10 rows is the 10 years of a simulation
    # then we save the full matrix -- Seyed will calculate on how to extract data

    # to index into the simulation results
    _c = [1, 2, 3] # carriage 1 2 3 
    _v = [1, 2]  # vaccine 1 = incidence vaccine, 2 = incidence no vaccine
    for c in _c 
        for v in _v 
        fname = "./output/$(simident)_sim$(nsim)_time$(ntimeyrs)_car$(c)_vac$(v)_$(fprefix).csv"
        # s is each simulation object which has two entries: [1]: prevalence,  [2]: incidence
        # for each incidence object s[2], 
        # it's a 4 dimensional object 
        
        inc_extract = reduce(vcat, [s[2][:, :, c, v] for s in res]) #s[2] is the incidence tensor
        writedlm(fname, inc_extract, ',')
        end 
    end   
    return scr # return the prevalance data for easy plotting 
end

function plot(res) 
    xvals = 1:size(res)[1]
    carc = res[:, 1:3] 
    carw = res[:, 4:6]
    cary = res[:, 7:9]
    
    println("last line: $(res[end, :])")

    #Gnuplot.options.default = sid
    @gp "reset"
    @gp :-  "set style fill transparent solid 0.5"
    #@gp :- "load '~/gpconfig.cfg'"
    @gp :- xvals carc[:, 2] carc[:, 3] "with filledcu notitle lc rgb '#e41a1c'"
    @gp :- xvals carw[:, 2] carw[:, 3] "with filledcu notitle lc rgb '#377eb8' "
    @gp :- xvals cary[:, 2] cary[:, 3] "with filledcu notitle lc rgb '#4daf4a' "
    
    @gp :- xvals carc[:, 1] "with lines title 'C' lc  rgb '#e41a1c' lw 2"
    @gp :- xvals carw[:, 1] "with lines title 'W' lc  rgb '#377eb8' lw 2"
    @gp :- xvals cary[:, 1] "with lines title 'Y' lc rgb '#4daf4a' lw 2"
    display(@gp)
end


function searchspace() 
    # this was a calibration effort to see the effect of beta 
    # we want transmission so that the steady state is 
    # the reported prevalence as reported in the Lancet paper
    adj1 = [0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
    adj2 = [0.38, 0.39, 0.4]
    adj3 = [0.5, 0.51, 0.52, 0.53, 0.54, 0.55]
    
    res = zeros(Float64, length(adj1)*length(adj2)*length(adj3), 7)
    t = 1
    for a1 in adj1
        for a2 in adj2 
            for a3 in adj3 
                cc = launch_sims(400, 3000, 0.001, a1, a2, a3)
                diffc = cc[end, 1] - cc[1, 1]
                diffw = cc[end, 4] - cc[1, 4]
                diffy = cc[end, 7] - cc[1, 7]
                res[t, :] .= (a1, a2, a3, diffc, diffw, diffy, diffc+diffw+diffy)
                t += 1
            end 
        end 
    end
    writedlm("calibration.dat", res, ',')
    return res
end