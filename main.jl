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


function launch_sims(nsims, startyear, endyear, base=0.00105, adj1=0.67, adj2=0.60, adj3=0.76)
    # new calibrated values=base=0.0011, adj1=0.65, adj2=0.49, adj3=0.73 
    # beta values (after calibration, but fitting to imd data 1997 to 2004)
    # beta values = (0.00105, 0.67, 0.60, 0.76)
    # note: this is with a cap of 2 (plus initial reset of cap at simulation start)
    @info "Current process id: $(myid())"
    @info "Total number of processors: $(nprocs())"
    @info "Number of simulations: $nsims"

    # create an object for model parameters
    mp = ModelParameters()
    mp.sstime = 0
    mp.beta = [base*adj1, base*adj2, base*adj3] # set the beta values for the disease
    mp.adj_vax_cov = true # false for basecase
    mp.cap_value = 2 # set to large value for calibration purposes 
    println("Model parameters: $(mp)")

    # transfer/initialze the parameters over all the workers 
    # each process has its own `p` modelparameters
    @everywhere init_state($mp) # initialize state/parameters over all processors

    cd = pmap(1:nsims) do x
        @info "Starting simulation $x on host $(gethostname()), id: $(myid())"
        init_agents(x, true) # true for reading calibrated data, false for new population
        #init_disease() # if not reading from calibrated files
        run_model(x, startyear, endyear, false)
    end
    @everywhere flush(stdout)
    println("simulations finished - returning data")
    return cd
end

function process_carriage(res, writefile=false)
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
    writefile && writedlm("./output/$(simident)_sim_avgprev_$(fprefix).csv", scr, ',')
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

function process_sims(res, fprefix="") 
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

function fit_imd(res, sc=0.7, sw=1.35, sy=0.55) 
    # res is the object returned from launch_sims() 
    # i.e. it's a vector of simulation results 
    # each result it self it a tuple(2) 
    # where x[1] is the carriage object 
    # and x[2] is the incidence object
    # In this function we convert simulation carriage, and calculate probability of IMD 
    # based on 1997 (or some year between 1997 and 2004)

    # calibrate the beta value so it fits to IMD cases from 1997 to 2004
    USPOP = 271394417 # 1997
    USUNIT = USPOP / 100000
    IMDTOTALS = [881, 21, 1110]
    IMDTOTALS_2004 = [203, 24, 253]
    IMDTOTALS_2000 = [442, 83, 699]
    
    # each incidence object (say res[i][2] where i is a simulation) 
    # is a tensor (4 dimensional): [time (in years), ag, carriage, vaccine]

    # for each simulation (of those who are not vaccineted -- though there should be zero people who are vaccinated)
    # get the total number of carraige in 2007 per serogroup, over all age groups 
    # convert it to total US population 
    # divide with the IMD to get IMD probability 

    # x[2] is the incidence object of each simulation (for each x simulation in res )
    # first, dims=4 adds the incidence from vaccinated/not vaccinated groups (but from 1997 to 2004, there is no one in the not-vaccinated group)
    # then we want to add accross age groups, which is dims=2, of the previous object
    # the reduction brings it to 8×500×3×1 Array{Int64, 4}:
    inc_peryear_perinf = reduce(hcat, [sum(sum(x[2], dims=4), dims=2) for x in res])
    inc_peryear_perinf = dropdims(inc_peryear_perinf, dims=4) # drop the "vax/novax dimension"
    
    # get the number of years in the simulation -- use the incidence object 
    LOY, _, _ = size(inc_peryear_perinf) # time x simulations x inftype 

    # the index [1, :, :, :] gets only the first row (i.e. 1997) of every simulation
    # take the mean of all the simulations
    # and convert to US population 
    firstyear = vec(mean(inc_peryear_perinf[1, :, :], dims=1) .* USUNIT)
    
    # divide by IMD (be careful to transpose it so column 1 is divided by the first element of IMD, etc )
    prob =  [sc, sw, sy] .* IMDTOTALS ./ firstyear
    println("prob: $prob")

    # Now go through each simulation and get the average 
    # number of carriage per year (average over simulations)
    # so we can flip a coin to see how many of them developed IMD 
    _yearly_avg = mean(inc_peryear_perinf, dims=2) # mean over the second dim will keep the carriage dimension open
    yearly_avg = round.(Int64, reshape(_yearly_avg, (LOY, 3)) .* USUNIT) # 8 years x 3 inf types 

    # for each column, flip that many coins to see how many IMD 
    c_flips = map(x -> sum(rand(x) .< prob[1]), yearly_avg[:, 1])
    w_flips = map(x -> sum(rand(x) .< prob[2]), yearly_avg[:, 2])
    y_flips = map(x -> sum(rand(x) .< prob[3]), yearly_avg[:, 3])
    imd_over_year = hcat(c_flips, w_flips, y_flips)
    return imd_over_year
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
    # total_imd_data = vec(sum(imd_data, dims=1))
    # println("total imd data: $(sum(total_imd_data))")
    # println("total imd sims: $(sum(total_imd_sims))")
    # lse = sum(abs.(total_imd_data .- total_imd_sims))
    # println("sum of diff: $lse")
    @gp "reset" 
    #@gp :- total_imd_data "with points pt 7 lc 'red'"
    @gp :- total_imd_sims "with lines lw 2 lc 'red'"
    display(@gp)
end