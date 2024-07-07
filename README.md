# Agent Based Model for IMD 
### How to Run
1. Initialization functions: `init_state()`, `init_agents()` 
2. Time Step function: `time_step()`

### Details, Assumptions, and Notes
`age_dynamics`: the function that simulates the age dynamics of the population, i.e. aging, births, and deaths in the population. There are two main dynamics: 
1. Suppose, in a time step, there are 50 agents to transition from <1 to 1-4. In this case, we need to find 50 agents to 'kill' in the population to maintain the population size. Note, also there is natural death for individuals past 100 years of age (and suppose there are 5 of them). Therefore, we will find 45 random ones (randomly selected) + the 5 from natural death and convert their IDs to newborns. 

2. This dynamic means we have to run the model for a while to get to equilibrium where the first age-group is constant. We ran for 10000 weeks and used the distribution at the end of 10000 weeks to initialize the model (see `init_agents()`).
