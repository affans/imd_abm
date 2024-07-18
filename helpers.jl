### helper, internal functions 

# corresponds to <1, 1--4, 5--10, 11--15, 16--23, 24--49, 50--64, 65++
const AG_BRAC = [0:51, 52:207, 208:519, 520:779, 780:1195, 1196:2547, 2548:3327, 3328:5252] 
# end should be 5251 but we add 1 more because of the way timestep() and age_dynamics() work 
# (the natural death happens after finding the age group -- and if they increase age to 5252, it will error)

function get_age_distribution() 
    # gets the age distribution of the population
    _ret = countmap([findfirst(Base.Fix1(∈, y.age), AG_BRAC) for y in humans])
end

function get_detailed_age_distribution() 
    # gets the age distribution of the population
    _ret = countmap([convert_week_to_year(y.age) for y in humans])
    sort(collect(_ret), by = x->x[1])
end

function get_disease_prop() 
    # gets the initial disease distribution of the population
    res = zeros(Int64, length(AG_BRAC), 6+1) #  +1 for total number of pop (without calling get_age_distribution()) 
    # columns: total pop size, not infected, infect C, infect Y, infect W
    for x in humans
        res[x.ag, Int(x.inf)] += 1 
        res[x.ag, 7] += 1 # gets the total number of people in that age group
    end
    # add 8th column
    res = hcat(res, vec(sum(res[:, 2:4], dims=2)))
    return res
end

function get_coverage_distribution() 
    # find 11-12 age group
    # find 11-15 age group 
    cc0 = length(findall(x -> x.age ∈ 572:623, humans))
    cc2 = length(findall(x -> x.age ∈ 832:883, humans))

    first0 = length(findall(x -> x.age ∈ 572:623 && x.vac == 1, humans))
    second0 = length(findall(x -> x.age ∈ 572:623 && x.vac == 2, humans)) # should always be zero

    # first2 will have a decent amount of people, but not a lot of them will have 
    # efficacy since it would've been expired... check with && x.eff > 0 condition
    first2 = length(findall(x -> x.age ∈ 832:883 && x.vac == 1, humans)) 
    second2 = length(findall(x -> x.age ∈ 832:883 && x.vac == 2, humans)) 

    (first0/cc0, second0/cc0, first2/cc2, second2/cc2)
end

@inline convert_year_to_wkrange(x::Int64) = (x*52):(x*52)+51 # convert year to week range
@inline convert_week_to_year(x::Int64) = floor(Int64, x/52)

function infoid(id)
    # an agent information function
    x = humans[id]
    agt = x.age == 52
    nd = x.age == 5201
    println("x.idx: $(x.idx) x.age: $(x.age), <1 -> 1-4? $agt, nat death: $nd")
end

### ONE-TIME USE / DEPCRECATED

function population_age_equilibrium(timetorun) 
    # this function was only used one to run the model for a certain time period
    # with only age dynamics to get the movement of the population aging through the system 
    # the CSV file was only used for visualization purposes and debugging 
    # I just used the print statement and copied the distribution to the init function

    # this function was run July 7th to get the population at the end of 10000 weeks 
    # and then the age distribution was used to initialize the population in the init_agents() function
    # note: this function may not be useful (or even work) as the model grows
    
    # steps to run this function for non-equilibrium age distribution
    # run init_state() to initialize the population
    # run init_agents() to initialize the agents with age distribution 
    res = zeros(Int64, timetorun, 8) # results 
    for i = 1:timetorun
        # this code is a subset of the code in timestep() 
        for x in humans
            x.age += 1
        end
        age_dynamics()
        distr = get_age_distribution()
        #display(distr)
        res[i, :] .= collect(values(distr))
    end
    #writedlm("population_age_equilibrium.csv", res, ',')
    println(getindex.(get_detailed_age_distribution(), 2))
    return res
end