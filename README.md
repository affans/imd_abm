# Agent Based Model for IMD 
### How to run
1. Initialization functions: `init_state()`, `init_agents()` 
2. To initialize disease prevalence: `init_disease()`
3. Time Step function: `timestep()`. A few functions run every time step (they can be independently run to test as well as long as the population is initialized)
   1. Age is incremented 
   2. Natural history of disease is applied by `naturalhistory()`. 
   3. Vaccine dynamics are run (essentially checked if efficacy has expired, and if so resetting those field). Note that this function is does not implement the administration of vaccine
   4. Transmission of disease is run by `transmission()`. 
   5. The logic of both `transmission`, `naturalhistory` (and even the initialization of disease function) uses 'swaps'. The swap sets the compartment the agent will move into, set by either initialization, transmission, or naturalhistory. 
4. There is a yearly time-loop as well which starts in the year '2005'. The `init_vaccine()` function runs everyyear to distribute vaccine. See details in the function for logic.

#### Example run commands 
```
julia> using Distributed
julia> using Revise
julia> includet("model.jl")
julia> init_state()
julia> simulate(1)
```

### Details, Assumptions, and Notes
`age_dynamics`: the function that simulates the age dynamics of the population, i.e. aging, births, and deaths in the population. There are two main dynamics: 
1. Suppose, in a time step, there are 50 agents to transition from <1 to 1-4. In this case, we need to find 50 agents to 'kill' in the population to maintain the population size. Note, also there is natural death for individuals past 100 years of age (and suppose there are 5 of them). Therefore, we will find 45 random ones (randomly selected) + the 5 from natural death and convert their IDs to newborns. 

2. This dynamic means we have to run the model for a while to get to equilibrium where the first age-group is constant. We ran for 10000 weeks and used the distribution at the end of 10000 weeks to initialize the model (see `init_agents()`).


`naturalhistory(x)` implements the natural history of disease. 

`transmission()` simulates transmission of disease using contact matrices 

`init_vaccine()` run yearly and distribute vaccines. 