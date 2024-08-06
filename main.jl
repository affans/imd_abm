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
            addprocs(SlurmManager(500), N=16, verbose="", topology=:master_worker, exeflags="--project=$(Base.active_project())")
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

function launch_sims(nsims, startyear, endyear, adj_vac_opt, adj_vac_cov, adj_vac_age, base=0.00105, adj1=0.67, adj2=0.60, adj3=0.76)
    # parameter options for the results (first draft)
    # nsims = 500, startyear=1997, endyear=2035 
    # adj_vac_opt = ("baseline", "r1", "r2") 
    # adj_vac_cov = coverage for cov2 (irrelevant for baseline -- since cov2=0.61 fixed)
    # adj_vac_age = age for cov2 (irrelevant for baseline -- since age=16 fixed)
    
    # calibration to steady state=base=0.0011, adj1=0.65, adj2=0.49, adj3=0.73 
    # calibration to IMD (following steady state): (0.00105, 0.67, 0.60, 0.76)
    # note: this is with a cap of 2 (plus initial reset of cap at simulation start)
    @info "Current process id: $(myid())"
    @info "Total number of processors: $(nprocs())"
    @info "Number of simulations: $nsims"

    # create an object for model parameters
    # if adj_vac_opt == "baseline" then adj_vac_cov, adj_vac_age are not relevant 
    # as they are not used in the `init_vaccine` function for baseline scenario
    mp = ModelParameters()
    mp.beta = [base*adj1, base*adj2, base*adj3] # set the beta values for the disease
    mp.adj_vax_opt = adj_vac_opt # vaccine option: baseline, r1, r2
    mp.adj_vax_cov = adj_vac_cov # coverage for cov2 
    mp.adj_vax_age = adj_vac_age # age for cov2
    mp.cap_value = 2 # set to large value for calibration purposes 
    @info "Model parameters: $(mp)"

    # transfer/initialze the parameters over all the workers 
    # each process has its own `p` modelparameters
    @everywhere init_state($mp) # initialize state/parameters over all processors

    @info "Starting Simulations -- redirecting to logfile"
    cd = pmap(1:nsims) do x
        @info "Starting simulation $x on host $(gethostname()), id: $(myid())"
        init_agents(x, true) # true for reading calibrated data, false for new population
        #init_disease() # if not reading from calibrated files
        run_model(x, startyear, endyear, false)
    end
    @everywhere flush(stdout)
    
    # comment this out if you don't want the simulations written to file
    strcov2 = string(round(Int, adj_vac_cov*100))
    if adj_vac_opt == "baseline"
        strsuffix = "baseline"
    else 
        strsuffix = "age$(adj_vac_age)_cov$(strcov2)_$(adj_vac_opt)"
    end
    process_carriage(cd, writefile=true, fsuffix=strsuffix) 
    process_sims(cd, writefile=true, fsuffix=strsuffix)
    fit_imd(cd, writefile=true, fsuffix=strsuffix)
    println("simulations finished - saving / returning data")
    return cd
end

function process_carriage(res; writefile=false, fprefix="aa", fsuffix="baseline")
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
    writefile && writedlm("./output/prevalence_$(fsuffix).csv", scr, ',')
    return scr
end

function plot_carriage(res) 
    # needs the results from process_carriage() 
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

function process_sims(res; writefile=false, fprefix="aa", fsuffix="baseline")
    # function to write the 4 dimensional incidence object to file for further analysis 

    # the resulting object has dimensions 
    # sim x time x agegroup x carriage x vaccine 
    # we want to split carriage/vaccine into it's own files 
    # and each file will have (time _ sim) x agegroup (sims are stacked on top of each other)

    # sim ident 
    simident = "r1"

    # get length/size inforamation
    nsim = length(res) 
    ntimeyrs, _, _, _ = size(res[1][2])  # since res[1][2] is the incidence matrix of the 1st simulation
 
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
        fname = "./output/incidence_car$(c)_vac$(v)_$(fsuffix).csv"
        # s is each simulation object which has two entries: [1]: prevalence,  [2]: incidence
        # for each incidence object s[2], 
        # it's a 4 dimensional object 
        
        inc_extract = reduce(vcat, [s[2][:, :, c, v] for s in res]) #s[2] is the incidence tensor
        writedlm(fname, inc_extract, ',')
        end 
    end   
    return
end

function fit_imd(res, sc=0.7, sw=1.35, sy=0.55; writefile=false, fprefix="aa", fsuffix="baseline")
    # res is the object returned from launch_sims() 
    # i.e. it's a vector of simulation results 
    # each result it self it a tuple(2) 
    # where x[1] is the carriage object 
    # and x[2] is the incidence object
    # In this function we convert simulation carriage, and calculate probability of IMD 
    # based on 1997 (or some year between 1997 and 2004)
    # To fit to IMD, run  xx = launch_sims(500, 1997, 2004, "baseline", 0.90, 17);

    nsim = length(res) 
    ntimeyrs, _, _, _ = size(res[1][2])  # since res[1][2] is the incidence matrix of the 1st simulation
 
    # calibrate the beta value so it fits to IMD cases from 1997 to 2004
    USPOP = 271394417 # 1997
    USUNIT = USPOP / 100000
    IMDTOTALS = [881, 21, 1110] ./ USUNIT
    
    # each incidence object (say res[i][2] where i is a simulation) 
    # is a tensor (4 dimensional): [time (in years), ag, carriage, vaccine]

    # for each simulation (of those who are not vaccineted -- though there should be zero people who are vaccinated)
    # get the total number of carraige in 2007 per serogroup, over all age groups 
    # convert it to total US population 
    # divide with the IMD to get IMD probability 

    # x[2] is the incidence object of each simulation (for each x simulation in res )
    # first, dims=4 adds the incidence from vaccinated/not vaccinated groups 
    # (but from 1997 to 2004, there is no one in the not-vaccinated group)
    # then we want to add accross age groups, which is dims=2, of the previous object
    # the reduction brings it to 8×500×3×1 Array{Int64, 4}:
    inc_peryear_perinf = reduce(hcat, [sum(sum(x[2], dims=4), dims=2) for x in res])
    inc_peryear_perinf = dropdims(inc_peryear_perinf, dims=4) # drop the "vax/novax dimension"
    
    # the index [1, :, :, :] gets only the first row of every simulation (which should be 1997)
    # take the mean of all the simulations
    # and convert to US population 
    firstyear = vec(mean(inc_peryear_perinf[1, :, :], dims=1))
    
    # divide by IMD to get the probability of IMD per infection state
    prob =  [sc, sw, sy] .* IMDTOTALS ./ firstyear

    # Now go through each simulation, inftype, age group, and vaccine status  
    # and for each incidence in these sub groups, flip a coin 
    eff_against_imd = [Beta(18.687,5.582), Beta(8.723, 7.809), Beta(2.603, 2.501)]
    Random.seed!(1248)
    all_sim_incidence = [x[2] for x in res] # incidence objects from the simulation 
    all_sim_imd = []
    for (i, sim_inc) in enumerate(all_sim_incidence) # for each simulation which gives a [time, age, inftype, vaccine]
        sim_imd = similar(sim_inc, Float64) # create a structure similar to sim_inc (ie. time x age x carriage x vaccine) to store IMD cases
        time, ag, car, vax = axes(sim_imd) # get the axes for each dimension 
        for ic in [1, 2, 3] # carriage 1 2 3 
            for vt in [1, 2] # vaccine 1 = incidence vaccine, 2 = incidence no vaccine
                samp_efficacy = [rand(eff_against_imd[1]), rand(eff_against_imd[2]), rand(eff_against_imd[3])]
                inc_cnts = sim_inc[:, :, ic, vt] # get the carriage incidence counts (time x age)
                imd_samples = map(inc_cnts) do x                     
                    efficacies = vt == 1 ? (1 - samp_efficacy[ic]) : 1 # check for efficacy
                    probs = prob[ic] .* efficacies
                    x * probs
                end
                sim_imd[:, :, ic, vt] .= imd_samples
            end
        end
        push!(all_sim_imd, sim_imd)
    end

    # write the data to a file -- same as writing incidence 
    if writefile 
        for c in [1, 2, 3]  # carriage 1 2 3 
            for v in [1, 2] # vaccine 1 = incidence vaccine, 2 = incidence no vaccine
                fname = "./output/imd_car$(c)_vac$(v)_$(fsuffix).csv"
                inc_extract = reduce(vcat, [s[:, :, c, v] for s in all_sim_imd])
                writedlm(fname, inc_extract, ',')
            end 
        end   
    end

    # add up over vaccine type, over age, and then take average over simulations 
    # dropdims(mean(dropdims(reduce(hcat, [sum(sum(z, dims=4), dims=2) for z in all_sim_imd]), dims=4), dims=2), dims=2) .* 271394417 / 100000
    return all_sim_imd
    #return round.(Int64, all_sim_imd)
end

function plot_imd(imd_obj)
    imd_data = [881	685	564	442	545	396	411	203 217 240 236 212 128 157 138 106 105 76 57 99 86 90 85 54 73 107;
    21 58 9 83 27 55 37 24 14 28 21 47 53 42 56 45 56 49 38 34 25 17 40 15 7 12;
    1110 810 641 699 621 391 372 253 250 251 247 292 252 231 205 124 109 70 46 52 30 48 68 50 28 59;
    ]

    println("size data: $(size(imd_data)), size sims: $(size(imd_obj))")
    println("sum data: $(sum(imd_data, dims=2))")
    println("sum sims: $(sum(imd_obj, dims=1))")

    # imd_obj will be three columns for each inf type, and time for rows
    # @gp "reset" 
    # @gp :- imd_obj[:, 1] "with lines title 'IMD C' lc 'red'"
    # @gp :- imd_obj[:, 2] "with lines title 'IMD W' lc 'green'"
    # @gp :- imd_obj[:, 3] "with lines title 'IMD Y' lc 'blue'"

    # @gp :- imd_data[1, :] "with points title 'Data C' pt 7 lc 'red'"
    # @gp :- imd_data[2, :] "with points title 'Data W' pt 7 lc 'green'"
    # @gp :- imd_data[3, :] "with points title 'Data Y' pt 7 lc 'blue'"
    # display(@gp)
    
    total_imd_sims = vec(sum(imd_obj, dims=2))
    total_imd_data = vec(sum(imd_data, dims=1))
    println("total imd data: $(sum(total_imd_data))")
    println("total imd sims: $(sum(total_imd_sims))")
    # lse = sum(abs.(total_imd_data .- total_imd_sims))
    # println("sum of diff: $lse")
    @gp "reset" 
    #@gp :- total_imd_data "with points pt 7 lc 'red'"
    @gp :- total_imd_sims "with lines lw 2 lc 'red'"
    display(@gp)
end