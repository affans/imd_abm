# main code to run, plot, and analyze large scale simulations 
# run this by `include("main.jl")` on the julia REPL (and on the headnode)
# or if using Revise, `using Revise; includet("main.jl")` to track changes 
# this file spawn workers and include the model on all workers 
# for dev, change to @everywhere includet(model.jl) to track changes (will require @everywhere using Revise)
# headnode code 

# main environment files 
using Gnuplot # maybe in master environment
using ClusterManagers 
using Distributed

ENV["JULIA_WORKER_TIMEOUT"] = 120  ## julia 1.8 is taking longer to connect to workers

# setup parallel workers
if nprocs() == 1
    if gethostname() == "hpc" 
        @info "connecting to hpc"
        addprocs(SlurmManager(450), N=15, verbose="", topology=:master_worker, exeflags="--project=$(Base.active_project())")
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

function launch_sims(nsims, base, adj1, adj2, adj3)
    @info "Current process id: $(myid())"
    @info "Total number of processors: $(nprocs())"
    @info "Number of simulations: $nsims"

    # create an object for model parameters
    mp = ModelParameters() 
    mp.maxtime = 6
    mp.beta = [base*adj1, base*adj2, base*adj3] # set the beta values for the disease
    println("Model parameters: $(mp)")

    # transfer/initialze the parameters over all the workers 
    # each process has its own `p` modelparameters
    @everywhere init_state($mp) # initialize state over all processors

    cd = pmap(1:nsims) do x
        simulate(x) # each simulation returns a matrix
    end

    # this code may change depending on output from simulate
    # cd is an array of matrix, extract each row (representing a variable) 
    # and construct their individual objects
    # after stacking, each object will be matrix time x sims (rows x cols)
    carc = reduce(hcat, [x[1, :] for x in cd])
    carw = reduce(hcat, [x[2, :] for x in cd])
    cary = reduce(hcat, [x[3, :] for x in cd])

    # take average of each variable over all simulations, these will be a vector of length maxtime
    carc_avg = mean(carc, dims=2)
    carw_avg = mean(carw, dims=2)
    cary_avg = mean(cary, dims=2)

    # repackage as a matrix to save as csv 
    car_avg = hcat(carc_avg, carw_avg, cary_avg)

    return car_avg
end

function plot(res) 
    @gp "reset" 
    @gp :- "load '~/gpconfig.cfg'"
    @gp :- res[:, 1] "with lines title 'C' lc 'black' lw 2"
    @gp :- res[:, 2] "with lines title 'W' lc 'blue' lw 2"
    @gp :- res[:, 3] "with lines title 'Y' lc 'red' lw 2"
    display(@gp)
end