# Agent Based Model for IMD 

### Scenarios


### How to run
First, read the model file
```
julia> using Distributed
julia> using Revise # for interactive changes
julia> includet("model.jl")
```
After model is loaded, 
1. Initialize the arrays: `init_state()`
2. Initialize the agents by `init_agents(simid, read_sims)` (sets up the demographics) and `init_disease()` (seeds the model with initial prevalence).  
   - The default of `init_agents() = init_agents(1, false)` initializes a 'clean' model. If running a clean model, follow with `init_disease` to seed prevalence. 
   - If `read_sims` is `true` then the model is initialized with a 'calibrated' system (see Calibration section). If running 500 simulations, you need 500 calibrated files. No need to run `init_disease` if using calibrated system. 

      NOTE: For reproducibility, download the calibrated model files  available online [here]. Adjust the file path in `init_agents()` accordingly. Alternatively, one can start a clean state and run until steady state by `init_agents(); init_disease()` (See Calibration section).
    
3. (Optional): Run `init_vaccine(cov1, cov2)` to vaccinate individuals with coverage values `cov1` and `cov2`. The value `cov1` will **always** apply to 11 year olds. The value `cov2` can be adjusted to apply to 15 year olds (or other ages in scenario analysis). If `cov1 = 0` then `cov2` coverage of 15 year olds will mean they are getting their **first** dose. The dynamics of `cov2` can be adjusted by `p.adj_vax_cov` (true/false) and `p.adj_vax_val` (Float64 value for `cov2`).
   - if `p.adj_cov_cov = true` then `cov1` is set to zero and `cov2` will dictage the age group that would get their **first** dose (since no one got their first dose at 11 years of age). 

4. Time Step function: `timestep()`. A few functions run every time step (they can be independently run to test as well as long as the population is initialized)
   1. Age is incremented 
   2. Natural history of disease is applied by `naturalhistory()`. 
   3. Vaccine dynamics are run (essentially checked if efficacy has expired, and if so resetting those field). Note that this function is does not implement the administration of vaccine
   4. Transmission of disease is run by `transmission()`. 
   5. The logic of both `transmission`, `naturalhistory` (and even the initialization of disease function) uses 'swaps'. The swap sets the compartment the agent will move into, set by either initialization, transmission, or naturalhistory. 

The function `run_model` combines all these methods together to run the model from start to end, with given years. 

#### Example run command (assuming running a single realization)  
```
julia> using Distributed
julia> using Revise
julia> includet("model.jl")
julia> init_state(); init_agents(); init_disease()
julia> run_model(1, startyear, endyear)
# startyear and endyear don't matter if not looking at vaccine
```

Or if running in a multi-core, distributed parallel (see `main.jl`)
```
mp = ModelParameters()
mp.beta = [base*adj1, base*adj2, base*adj3] # set the beta values for the disease
mp.adj_vax_cov = adj_vac_cov # false for basecase
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
```

### Calibration 

### Scenario Analysis
#### Option 1 (R1) 
In `main.jl`, we set up scenarios in the main function. The **baseline** scenario is always fixed to `cov1=90%`, `cov2=60%` where `cov1` is the coverage of 11 year olds and `cov2` is the coverage for 16 year olds getting their second dose. 

In Option 1 `p.adj_vax_opt = "r1"`, then `cov1 = 0` after 2025. The variable `cov2` (adjusted by `adj_vax_cov`) then signifies the coverage of individuals (of age `adj_vax_age`) getting vaccinated. This coverage will prioritize those getting their second dose, but eventually will start giving the age group their first dose since `cov1 = 0`. The value of `cov2` is set by . 

See table for summary

```
+=============+===========+==========+========+==+
| Baseline Scenario       |          |        |  |
+=============+===========+==========+========+==+
|             | cov1=90%  | cov2=61% | age=16 |  |
+-------------+-----------+----------+--------+--+
| Adjusted Scenarios      |          |        |  |
+-------------+-----------+----------+--------+--+
| age  / cov2 | 15        | 16       | 17     |  |
+-------------+-----------+----------+--------+--+
| 61%         | ?         | ?        | ?      |  |
+-------------+-----------+----------+--------+--+
| 90%         | ?         | ?        | ?      |  |
+-------------+-----------+----------+--------+--+
note: adjusted scenario means `cov1 = 0`
```




### Details, Assumptions, and Notes
`age_dynamics`: the function that simulates the age dynamics of the population, i.e. aging, births, and deaths in the population. There are two main dynamics: 
1. Suppose, in a time step, there are 50 agents to transition from <1 to 1-4. In this case, we need to find 50 agents to 'kill' in the population to maintain the population size. Note, also there is natural death for individuals past 100 years of age (and suppose there are 5 of them). Therefore, we will find 45 random ones (randomly selected) + the 5 from natural death and convert their IDs to newborns. 

2. This dynamic means we have to run the model for a while to get to equilibrium where the first age-group is constant. We ran for 10000 weeks and used the distribution at the end of 10000 weeks to initialize the model (see `init_agents()`).


`naturalhistory(x)` implements the natural history of disease. 

`transmission()` simulates transmission of disease using contact matrices 

`init_vaccine()` run yearly and distribute vaccines. 

In scenario analysis, we want evaluate different vaccine scenarios 
by changing coverage values of fdc/ddc. The scenario is what happens if the 
first dose is not given at 11 years (i.e. set fdc coverage=0.0) and the first dose 
is shifted to the 16 year age group (i.e., coverage in ddc). 
In the base case scenario (where vaccine IS given to 11 year olds), 
the coverage (say 60%) should really only be from agents who had their first dose. 
In other words length(cc2_vax) > tv2 (so we can select tv2 people from cc2_vax array)
In scenario analysis, first dose of 11 year is stopped so at some point 
length(cc2_vax) becomes small and we can not select tv2 people. In this case, 
we will priority whoever is in cc2_vax and then select from the overall age group 