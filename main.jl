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

function run_all() 
    # Calibration procedure (last update aug 12)
    #    startyear, endyear = 1940 to 2000 
    #    betavalues: (0.0011, 0.65, 0.60, 0.68)
    #    init_agents(x, false) # false for new population 
    #    init_disease() # since not reading from calibrated files
    #    run_model(set writefiles parameter to true)
    #    Note: comment out the vaccine code to make it run faster?
    #    Note: the save_sim part is commented out in `model.jl` to prevent overwriting saved calibration files
    #    Note: with betas above, prevalence after 40 years C, W, Y (with uncertainty): [920.126, 689.0, 1151.25, 119.648, 0.0, 285.05, 935.042, 704.7, 1184.3500000000001]

    # Run procedure (last update aug 12)
    #    Beta values following steady state to calibrate to IMD (0.00104, 0.65, 0.58, 0.66)
    #    set mp.cap_value = 2
    #    init_agents(x, true) # true to read existing calibrated data
    #    #init_disease() # commented out
    #    run_model(x, startyear, endyear, false) # set kwarg to false to not overwrite the calibrated files

    # Note: The probabilities to calculate IMD are based on the output of running the model 
    # from 1995 to 2004 (can run it for longer, but it only uses for 1997 to 2004)
    # The probabilities are calculated using the `fit_and_plot_imd` function
    # See inside function for some parameters to set for the optimization procedure
    # Calibrated/optimized on August 12 (and the probs set as default values)

    calibrated_beta_values = (base=0.00104, adj1=0.65, adj2=0.58, adj3=0.66)
    xx = launch_sims(500, 1995, 2035, "baseline", 0.61, 16, calibrated_beta_values...);
    xx = launch_sims(500, 1995, 2035, "r1", 0.61, 15, calibrated_beta_values...);
    xx = launch_sims(500, 1995, 2035, "r1", 0.90, 15, calibrated_beta_values...);
    xx = launch_sims(500, 1995, 2035, "r1", 0.61, 16, calibrated_beta_values...);
    xx = launch_sims(500, 1995, 2035, "r1", 0.90, 16, calibrated_beta_values...);
    xx = launch_sims(500, 1995, 2035, "r1", 0.61, 17, calibrated_beta_values...);
    xx = launch_sims(500, 1995, 2035, "r1", 0.90, 17, calibrated_beta_values...);
    xx = launch_sims(500, 1995, 2035, "r2", 0.90, 17, calibrated_beta_values...);
    xx = launch_sims(500, 1995, 2035, "r2", 0.61, 17, calibrated_beta_values...);
    # r3 here, cov is not relevant but put it here for correct filename
    xx = launch_sims(500, 1995, 2035, "r3", 1.0, 16, calibrated_beta_values...);
    xx = launch_sims(500, 1995, 2035, "r3", 1.0, 17, calibrated_beta_values...);
end

function launch_sims(nsims, startyear, endyear, adj_vac_opt, adj_vac_cov, adj_vac_age, base=0.00104, adj1=0.65, adj2=0.58, adj3=0.66)
    # parameter options for the results (aug 12) 
    # nsims = 500, startyear=1995, endyear=2035 
    # adj_vac_opt = ("baseline", "r1", "r2") 
    # adj_vac_cov = coverage for cov2 (irrelevant for baseline -- since cov2=0.61 fixed)
    # adj_vac_age = age for cov2 (irrelevant for baseline -- since age=16 fixed)
   
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
    mp.cap_value = 2 
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
    
    # comment this out if you don't want the simulations written to file (e.g. calibration)
    strcov2 = string(round(Int, adj_vac_cov*100))
    if adj_vac_opt == "baseline"
        strsuffix = "baseline"
    else 
        strsuffix = "age$(adj_vac_age)_cov$(strcov2)_$(adj_vac_opt)"
    end
    process_carriage(cd, writefile=true, fsuffix=strsuffix) 
    process_sims(cd, writefile=true, fsuffix=strsuffix)
    process_imd(cd, writefile=true, fsuffix=strsuffix)
    println("simulations finished - saving / returning data")
    return cd
end

function fit_and_plot_imd(res) 
    # function uses a minimization procedure to fit the incidence data to IMD data 

    # IMD data from 1997 to 2024
    USPOP = 271394417 # 1997
    USUNIT = USPOP / 100000    
    imd_data = [881	685	564	442	545	396	411	203 217 240 236 212 128 157 138 106 105 76 57 99 86 90 85 54 73 107;
                21 58 9 83 27 55 37 24 14 28 21 47 53 42 56 45 56 49 38 34 25 17 40 15 7 12;
                1110 810 641 699 621 391 372 253 250 251 247 292 252 231 205 124 109 70 46 52 30 48 68 50 28 59;
                ] ./ USUNIT

    # x labels for the plot (account for the fact that sims may not start in 1995)
    startyear = 1995 # start year of the simulations (we want to fit to 1997 to 2004 only)
    stindx = 1997 - startyear 
    
    # fitting process
    sc=0.000040749466224307044 
    sw=0.000020140149943183043 
    sy=0.000033288114220056214
    function probsqerror(prob) 
        q = process_imd(res, prob[1], prob[2], prob[3])
        data_to_fit = imd_data[:, 1:8]' # from 1997 to 2004
        sims_to_fit = q[(stindx + 1):(stindx + 8), :]
        mabe = mean((data_to_fit .- sims_to_fit).^2)    
        #println("Mean error: $mabe")
        return mabe
    end
    opt = optimize(probsqerror, [sc, sw, sy])
    println(opt)
    println("Fitted probabilities: $(Optim.minimizer(opt))")
    # get the new imd obj with fitted values 
    imd_obj = process_imd(res, Optim.minimizer(opt)...)

    xsims = 1:size(imd_obj)[1] 
    xdata = (1:size(imd_data)[2]) .+ stindx
    

    # code to print the years for xtics
    # for x in xsims 
    #     print("'$(startyear + x - 1)' $(x), ")
    # end
    #return mabe
    # imd_obj will be three columns for each inf type, and time for rows
    @gp "reset" 
    @gp :- "set xtics  rotate 90 ('1995' 1, '1996' 2, '1997' 3, '1998' 4, '1999' 5, '2000' 6, '2001' 7, '2002' 8, '2003' 9, '2004' 10, '2005' 11, '2006' 12, '2007' 13, '2008' 14, '2009' 15, '2010' 16, '2011' 17, '2012' 18, '2013' 19, '2014' 20, '2015' 21, '2016' 22, '2017' 23, '2018' 24, '2019' 25, '2020' 26, '2021' 27, '2022' 28, '2023' 29, '2024' 30, '2025' 31, '2026' 32, '2027' 33, '2028' 34, '2029' 35, '2030' 36, '2031' 37, '2032' 38, '2033' 39, '2034' 40, '2035' 41)"
    @gp :- "set object rectangle from $(stindx + 1),graph 0 to $(stindx + 8),graph 1 fillcolor rgb 'gray' fillstyle transparent solid 0.5 noborder"
    @gp :- xsims imd_obj[:, 1] "with lines title 'IMD C' lc 'red'"
    @gp :- xsims imd_obj[:, 2] "with lines title 'IMD W' lc 'green'"
    @gp :- xsims imd_obj[:, 3] "with lines title 'IMD Y' lc 'blue'"

    @gp :- xdata imd_data[1, :] "with points title 'Data C' pt 7 lc 'red'"
    @gp :- xdata imd_data[2, :] "with points title 'Data W' pt 7 lc 'green'"
    @gp :- xdata imd_data[3, :] "with points title 'Data Y' pt 7 lc 'blue'"

    @gp :- xsims sum(imd_obj, dims=2) "with lines title 'IMD Total' lc 'black' lw 2"
    @gp :- xdata sum(imd_data, dims=1) "with points title 'Data Total' pt 7 lc 'black'"
    display(@gp)
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

function process_imd(res, sc=0.00004187, sw=0.000013548, sy=0.0000480388; writefile=false, fprefix="aa", fsuffix="baseline")
    # In this function we use the simulation carriage results, and calculate IMD cases based on given probability
    # res is the object returned from launch_sims(), i.e. it's a vector of simulation results 
    # each result it self it a tuple(2) where x[1] is the carriage object and x[2] is the incidence object
    
    # !!! default probabilities are based on optimization procedure `fit_and_plot_imd` which is run on the output of launch_sims()

    nsim = length(res) 
    ntimeyrs, _, _, _ = size(res[1][2])  # since res[1][2] is the incidence matrix of the 1st simulation

    # x[2] is the incidence object of each simulation (for each x simulation in res)
    # x[2] is a 4 dimensional object [time, age, carriage, vaccine]
    # sum over dims=4 adds the incidence from vaccinated/not vaccinated groups (from 1997 to 2004, there is no one in the not-vaccinated group)
    # sum over dims=2 adds the incidence over age groups since we want to look at total incidence per serogroup
    # then we drop the last dims to bring to 8×500×3: 8 time points, 500 simulations, 3 serogroups
    #inc_peryear_perinf = reduce(hcat, [sum(sum(x[2], dims=4), dims=2) for x in res])
    #inc_peryear_perinf = dropdims(inc_peryear_perinf, dims=4) # drop the "vax/novax dimension"
    #firstyear = vec(mean(inc_peryear_perinf[1, :, :], dims=1))
    #println("first year average of $nsim simulations: $firstyear")
    
    # divide by IMD to get the probability of IMD per infection state
    prob =  [sc, sw, sy] # .* IMDTOTALS ./ firstyear
    #println(prob)

    # Now go through each simulation, inftype, age group, and vaccine status  
    # and for each incidence in these sub groups, flip a coin 
    eff_against_imd = [Beta(18.68838313,5.58224431), Beta(5.525016302, 2.19696124), Beta(2.529281944, 2.430094417)]
    Random.seed!(482)
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

    # average over the simulations to give IMD cases (time x simulations x carriage) 
    _q = reduce(hcat, [dropdims(sum(sum(x, dims=4), dims=2), dims=4) for x in all_sim_imd])
    q = dropdims(mean(_q, dims=2), dims=2) # take the mean over simulations
    return q 

    # add up over vaccine type, over age, and then take average over simulations 
    # dropdims(mean(dropdims(reduce(hcat, [sum(sum(z, dims=4), dims=2) for z in all_sim_imd]), dims=4), dims=2), dims=2) .* 271394417 / 100000
    # return all_sim_imd
end

