using StatsBase, Random
using DelimitedFiles
using Distributions
# for saving calibration results
using JLD2 
using UnPack

# make sure the order doesn't change as CARC=2, CARW=3, ... is used in init_disease
@enum INFTYPE UNDEF=0 SUS=1 CARC=2 CARW=3 CARY=4 IMD=5 REC=6

# define an agent and all agent properties
# if adjusting this struct, make sure to adjust Human() as well
Base.@kwdef mutable struct Human
    idx::Int64 = 0
    age::Int64 = 0 # age in weeks
    ag::Int8 = 0 # store the age group for easier access to other properties
    inf::INFTYPE = SUS # infection status
    swap::INFTYPE = UNDEF # swap state
    p_inf::INFTYPE = UNDEF # previous infection state
    c_inf::Int16 = 0 # infection count
    tis::Int16 = 0 # time in state
    st::Int16 = typemax(Int16) # max time in state 
    vac::Int16 = 0 # dose count
    eff::Float64 = 0.0 # vaccine efficacy
    exp::Int64 = 0 # waning time of vaccine
end
Base.show(io::IO, ::MIME"text/plain", z::Human) = dump(z)

## system parameters
Base.@kwdef mutable struct ModelParameters
    beta::Vector{Float64} = [0.5, 0.5, 0.5] # for Carriage 
    prob_of_imd::Float64 = 0.0
    adj_vax_cov::Bool = false # flip to true for scenario analysis
    adj_vax_val::Float64 = 0.61 # coverage for the second dose (or the first dose if 11 year olds are not getting vaccinated)
    cap_value::Int16 = 99  # maximum number of infections
end
# constant variables
const humans = Array{Human}(undef, 0) 
const p = ModelParameters()  ## setup default parameters
const POPSIZE = 100000

# distributions 
const DISTR_CARRIAGE = Poisson(42) # From Feldman paper, 9.6 months ~ 42 weeks
const DISTR_RECOVER = Poisson(244)
const VAC_EFF = Beta(5.68, 2.1437)
const VAC_DURATION = Poisson(244) # seyed to go check

includet("helpers.jl")

### Main simulation functions
function run_model(simid, startyear, endyear, save_sim = false)
    # main time loop function 
    @info "Running simulation ID: $simid"
    @info "Model parameters: $(p)"
    @info "Simulation data written on file: $(save_sim)"
    
    # set seed to simulation id
    Random.seed!(simid*556)

    tm_years = startyear:endyear 
    ytm = repeat(tm_years, inner=52) # create an array with the years for indexing the year
    totaltime = length(ytm) # in weeks for the time loop 
    @info "tm_years: $(tm_years), total years: $(length(tm_years))" 
    @info "expanded to array size: $totaltime (total years * 52)" 
    @info "caution: there are no error checks on start/end date"
    startyear ∈ 2005:2035 && @info "start year set between 2005-2035, vaccine will happen"

    # data collection -- allocate memory 
    ss_inc = zeros(Int16, length(tm_years), length(AG_BRAC), 3, 2) # time(in years) x ag x 3carriage x 2vaccine
    ss_carriage = zeros(Int64, 6, totaltime) # data 3 rows for C, W, Y since

    @inbounds for t in 1:totaltime 
        currentyear = ytm[t]
        if t % 52 == 0 
            cov1, cov2 = get_coverage(currentyear) 
            init_vaccine(cov1, cov2)
        end 
        wv = timestep() 
        ss_inc[currentyear - (startyear - 1), :, :, :] += wv 
        ss_carriage[:, t] .= collect_data()
    end

    # STEADY STATE 
    # I used the function to run 20000 weeks (with no vaccine) to reach steady state in the population
    # At the end of each simulation, the `humans` array object is saved as a JLD2 file 
    # These files can be read again in simulations as initial conditions to avoid running the initial 3000 weeks. 
    # The steady state files are stored in the cluster and not checked into repo 
    
    # commenting this out since we don't want to be overwriting the calibration files 
    # if save_sim
    #     @info "saving simulation to jld2"
    #     file = jldopen("/data/imd_abm_calibration/sim$(simid)n.jld2", "w") 
    #     file["h"] = humans #save the humans object 
    #     close(file)
    # end
    return ss_carriage, ss_inc
end

function collect_data()
    # keep optimized since this runs in the time loop
    _s, _c, _w, _y, _i, _r  = 0, 0, 0, 0, 0, 0
    distr = countmap([x.inf for x in humans])   
    haskey(distr, SUS) && (_s = distr[SUS])
    haskey(distr, CARC) && (_c = distr[CARC])
    haskey(distr, CARW) && (_w = distr[CARW])
    haskey(distr, CARY) && (_y = distr[CARY])    
    haskey(distr, IMD) && (_i = distr[IMD])
    haskey(distr, REC) && (_r = distr[REC])
    return (_s, _c, _w, _y, _i, _r)
end

### Iniltialization Functions
function read_jld_calibration(id)
    filename = "/data/imd_abm_calibration/sim$(id)n.jld2"
    file = jldopen(filename, "r")
    obj = file["h"] 
    close(file)
    return obj
end

init_state() = init_state(ModelParameters())
function init_state(ip::ModelParameters)
    # the p is a global const
    # the ip is an incoming different instance of parameters 
    # copy the values from ip to p. 
    # ip.popsize == 0 && error("no population size given")
    for x in propertynames(p)
        setfield!(p, x, getfield(ip, x))
    end
    # resize the human array to change population size
    resize!(humans, POPSIZE)
    @debug "Initialization state complete..."
    return
end

init_agents() = init_agents(1, false)
function init_agents(simid, read_sims=false) 
    # non-equilibrium age distribution
    # pop_distr = [1102, 1077, 1118, 1137, 1163, 1187, 1218, 1229, 1225, 1221, 1223, 1236, 1262, 1289, 1338, 1335, 1312, 1300, 1308, 1296, 1289, 1312, 1313, 1292, 1290, 1293, 1302, 1325, 1353, 1374, 1405, 1426, 1430, 1389, 1365, 1349, 1352, 1350, 1322, 1341, 1338, 1318, 1311, 1257, 1229, 1213, 1171, 1183, 1160, 1175, 1231, 1304, 1283, 1234, 1207, 1210, 1228, 1274, 1304, 1303, 1306, 1302, 1277, 1252, 1239, 1214, 1159, 1130, 1088, 1040, 998, 948, 901, 873, 850, 891, 625, 595, 573, 573, 484, 427, 385, 351, 321, 279, 251, 224, 187, 167, 145, 125, 105, 84, 68, 53, 40, 30, 21, 15, 23]
    
    if read_sims # reads saved simulation file
        @info "Reading calibrated model files"
        h = read_jld_calibration(simid)
        humans .= h
        for x in humans 
            x.c_inf = 0
        end
        return 
    end
    @info "Initialzing new zero population"

    # equilibrium age distribution
    pop_distr = [1442, 1433, 1422, 1407, 1400, 1385, 1371, 1371, 1362, 1340, 1321, 1326, 1307, 1292, 1288, 1283, 1278, 1260, 1266, 1244, 1237, 1234, 1184, 1226, 1195, 1165, 1186, 1190, 1157, 1146, 1113, 1098, 1132, 1109, 1097, 1067, 1101, 1068, 1054, 1037, 1072, 1068, 1032, 1027, 1021, 1000, 979, 976, 940, 952, 958, 956, 958, 949, 933, 939, 916, 904, 868, 886, 919, 867, 892, 863, 834, 845, 873, 838, 804, 815, 813, 826, 823, 789, 806, 813, 810, 788, 779, 741, 732, 766, 713, 731, 706, 731, 712, 729, 724, 719, 667, 692, 693, 707, 652, 639, 661, 653, 648, 630, 629]
    agb = convert_year_to_wkrange.(0:100) # get the week range for each year
    sampled_ages = shuffle!(rand.(inverse_rle(agb, pop_distr))) # sample
    
    # error checks
    @assert sum(pop_distr) == POPSIZE
    @assert length(sampled_ages) == POPSIZE

    # if rounding issues, try this (from the rsv canada project) 
    # push!(sf_agegroups, fill(6, abs(p.popsize - length(sf_agegroups)))...) # if length(sf_agegroups) != p.popsize # because of rounding issues
    for i in 1:POPSIZE
        humans[i] = Human()
        x = humans[i]
        x.idx = i
        x.age = sampled_ages[i]
        x.ag = findfirst(Base.Fix1(∈, sampled_ages[i]), AG_BRAC)
    end
    return 
end

function init_disease()
    @info "Initializing disease..."
    ag_carriage_prev = rand.([Uniform(0.028, 0.085), Uniform(0.031, 0.091), 
                                Uniform(0.042, 0.118), Uniform(0.055, 0.174), 
                                Uniform(0.106, 0.334), Uniform(0.053, 0.264), 
                                Uniform(0.034, 0.125), Uniform(0.026, 0.090)])

    _sero_prop = [0.0, 0.0, 0.0] # Index: 1=C, 2=W, 3=Y -- ORDER MATTERS
    _sero_prop[1] = rand(Uniform(0.41, 0.47)) # SERO C
    _sero_prop[2] = rand(Uniform(0.02, 0.05)) # SERO ּW    
    _sero_prop[3] = 1 - _sero_prop[1] - _sero_prop[2] # SERO Y

    sero_pop = Categorical(_sero_prop)
    @info "...sero props $_sero_prop"

    # error check -- make sure population is clean
    findfirst(x -> (x.inf ≠ SUS || x.swap ≠ UNDEF), humans) ≠ nothing && error("Population not clean. Are you initializing from saved files?")

    for ag in 1:length(AG_BRAC)
        prev = ag_carriage_prev[ag]
        elig = findall(x -> x.ag == ag && rand() < prev, humans)
        for id in elig 
            x = humans[id]
            x.swap = INFTYPE(rand(sero_pop) + 1)  # since CARC = 2, CARY = 3, CARW = 4
            activate_swaps(x)
        end
    end
    @info "...Disease Initialized" get_disease_prop()
    return
end

@inline function get_coverage(year) 
    # helper function to get the vaccine coverage value for each year
    # only works for years between 2005 and 2025 (+ fixed values for 2026 - 2035) 
    # as this is hard coded
    cov1 = 0.0 
    cov2 = 0.0 
    
    # create coverage values from 2005 to 2024 (fixed vector of 20 elements)
    fdc = [0.13, 0.19, 0.35, 0.45, 0.55, 0.62, 0.67, 0.69, 0.73, 0.74, 0.75, 0.77, 0.79, 0.80, 0.81, 0.83, 0.86, 0.9, 0.9, 0.9]
    ddc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.06, 0.10, 0.14, 0.18, 0.23, 0.29, 0.33, 0.37, 0.41, 0.46, 0.49, 0.55, 0.61, 0.61, 0.61]
    @assert length(fdc) == length(ddc) == 20 

    if year ∈ 2005:2024
        cov1 = fdc[year - 2004]
        cov2 = ddc[year - 2004]
    end 
    if year > 2024 
        # coverage remains fixed for beyond 2024
        cov1 = 0.9
        cov2 = 0.61 
        if p.adj_vax_cov
            # we are in "adjusted" scenario analysis, 
            cov1 = 0.0 
            #cov2 = 0.61 # could also test at 90%
            cov2 = p.adj_vax_val
        end
    end 
    return cov1, cov2
end

function init_vaccine(cov1, cov2)
    # implementing vaccine every year starting in 2005
    # cov1, cov2 are coverage values for first dose and second dose

    cov1 + cov2 == 0 && return

    # find everyone 11 years of age 
    cc1 = shuffle!(findall(x -> x.age ∈ (572:623), humans))
    tv1 = round(Int64, cov1 * length(cc1)) # get total that will get vaccinated
    e1 = @view cc1[1:tv1] # view their indices
    for idx in e1  # vaccinate them 
        x = humans[idx]
        vaccinate(x)
    end

    # for second dose, find *everyone* 16 years of age and then multiply by coverage value
    # for other scenario: also do for 15 (832:883) and 17 (884:935) years of age
    cov2_agegroup = (832:883)
    cc2 = shuffle!(findall(x -> x.age ∈ cov2_agegroup, humans))
    tv2 = round(Int64, cov2 * length(cc2)) # total number of agents vaccinated

    # find agents who are eligble for second dose (i.e. got their first dose)
    # these would be the first agents to get the second dose -- and then select the rest from cc2
    cc2_vax = shuffle!(findall(x -> x.age ∈ cov2_agegroup && x.vac > 0, humans))

    # In scenario analysis, we want evaluate different vaccine scenarios 
    # by changing coverage values of fdc/ddc. The scenario is what happens if the 
    # first dose is not given at 11 years (i.e. set fdc coverage=0.0) and the first dose 
    # is shifted to the 16 year age group (i.e., coverage in ddc). 
    # In the base case scenario (where vaccine IS given to 11 year olds), 
    # the coverage (say 60%) should really only be from agents who had their first dose. 
    # In other words length(cc2_vax) > tv2 (so we can select tv2 people from cc2_vax array)
    # In scenario analysis, first dose of 11 year is stopped so at some point 
    # length(cc2_vax) becomes small and we can not select tv2 people. In this case, 
    # we will priority whoever is in cc2_vax and then select from the overall age group 

    # The code to do basecase and scenario analysis is the same 
    # in this case, cc2_vax should have enough people at the start so that 
    # when we do _elig[1:tv2] we only select agents from cc2_vax 
    # but I threw an assert jsut to be sure for the basecase
    !(p.adj_vax_cov) && @assert length(cc2_vax) >= tv2 
    _elig = append!(copy(cc2_vax), setdiff(cc2, cc2_vax)) # faster than splatting
    #_elig = [cc2_vax..., setdiff(cc2, cc2_vax)...] # too many allocations
    
    e2 = @view _elig[1:tv2]
    for idx in e2
        x = humans[idx]
        vaccinate(x)
    end

    #@info "year $year, first dose $cov1 second dose $cov2"
    #@info "total in group 11-12: $(length(cc1)), 16: $(length(cc2)) (with vax: $(length(cc2_vax)))"    
    #@info "total v1: $tv1, v2: $tv2" 
end

@inline function newborn(id)
    # converts a human to a newborn
    x = humans[id]
    x.age = 0
    x.ag = 1
    x.inf = SUS 
    x.swap = UNDEF
    x.p_inf = UNDEF 
    x.c_inf = 0
    x.tis = 0
    x.st = typemax(Int16)
    x.vac = 0 
    x.eff = 0.0
    x.exp = 0 
    return
end

### TIME STEP FUNCTIONS
function timestep() 
    # main timestep function -- advanced simulation by 1 week 
    # age dynamics - increase age by one and then run age dynamics

    # allocate 
    week_inc = zeros(Int16, length(AG_BRAC), 3, 2) # ag x carriage x vaccine
    for x in humans
        x.age += 1 
        x.ag = findfirst(Base.Fix1(∈, x.age), AG_BRAC)
        x.tis += 1 
        swaps = naturalhistory(x) # move through the natural history of the disease first         
        cf = activate_swaps(x) # activate any swaps, cf is a counter for only if swap to carriage
        if cf # if infection, recall CARC=2 CARW=3 CARY=4
            _vs = x.eff > 0 ? 1 : 2
            week_inc[x.ag, Int(x.inf) - 1, _vs] += 1 
        end
        check_vaccine_expiry(x) # see if vaccine efficacy has expired
    end
    transmission() # naturalhistory affects swaps of carriage and rec. Swap for carriage is irrelevant, swap might rec -> sus, but activate_swaps comes after, it doesn matter
    age_dynamics() 
    return week_inc
end

function vaccinate(x)
    x.vac += 1 # increase dose count
    x.eff = rand(VAC_EFF) # set efficacy 
    x.exp = x.age + rand(VAC_DURATION) # set expiry
    x.vac == 3 && error("someone got 3 doses.")
end

function naturalhistory(x::Human)
    cnt = 0 
    if x.tis == x.st # move through nat history of disease
        cnt += 1
        x.inf == SUS && error("SUS state can not expire")
        if x.inf ∈ (CARC, CARY, CARW)
            if rand() < p.prob_of_imd 
                x.swap = IMD
            else
                x.swap = REC
            end    
        end
        if x.inf == IMD 
            x.swap = REC
        end
        if x.inf == REC
            x.swap = SUS
        end
    end
    return cnt
end

function activate_swaps(x::Human)
    cc = false # if carriage happens
    if x.swap ≠ UNDEF 
        if x.swap in (CARC, CARW, CARY)
            move_to_carriage(x)
            cc = true
        elseif x.swap == IMD 
            move_to_imd(x)
        elseif x.swap == REC 
            move_to_rec(x)
        elseif x.swap == SUS 
            move_to_sus(x)
        end     
        x.tis = 0  # reset time in state
        x.swap = UNDEF # reset swap state
    end 
    return cc
end

@inline function move_to_carriage(x::Human) 
    x.inf = x.swap # so will be CARC, CARY, CARW
    x.c_inf += 1 # raise infection cnt
    x.st = round(Int16, rand(DISTR_CARRIAGE))     
end

@inline function move_to_imd(x::Human) 
    x.inf = IMD 
    x.st = 2 
end

@inline function move_to_rec(x::Human) 
    # before recover, record previous infection state
    x.p_inf = x.inf
    x.inf = REC
    x.st = Int16(rand(DISTR_RECOVER))
end

@inline function move_to_sus(x::Human)
    x.inf = SUS
    x.st = typemax(Int16)
end

function check_vaccine_expiry(x)
    # efficacy only lasts 5 - 6 years. Reset after time expires
    if x.exp == x.age 
        x.eff = 0.0 
        x.exp = 0 
    end
end

function age_dynamics() 
    # function deals with newborns and deaths

    # get the number of agents switch from <1 to 1-4
    num_to_die = length(findall(x -> x.age == 52, humans)) # only length because we need to select this many random
    # # find all individuals that will die due to natural death
    deathids = findall(x -> x.age == 5252, humans)

    # #println("total number of newborns to introduce: $num_to_die")
    # #println("total number of natural deaths: $(length(deathids))")

    # # need to a total of `num_to_die` including the natural deaths 
    # # so fill in the rest of deaths by randomly sampling from the population
    # # ensure that those transitioning to 1-4 and 100+ are not part of sampling
    eligible = findall(x -> x.age > 52 && x.age < 5200, humans)
    
    # # add to the death IDs randomly selected ones 
    for _ in 1:(num_to_die - length(deathids))
        push!(deathids, rand(eligible))
    end

    # turn them into newborns! Use infoid() to print statistics
    newborn.(deathids)
    return
end

function transmission() 
    len_agbrac = length(AG_BRAC)
    incidence = zeros(Int64, len_agbrac)
    
    inf_agents = humans[findall(x -> x.inf ∈ (CARC, CARW, CARY), humans)]
    #length(inf_agents) == 0 && return incidence # no carriers to transmit
    buckets = shuffle!.([findall(x -> x.ag == i, humans) for i = 1:len_agbrac])
    totalcontacts = 0.0 
    @inbounds for x in inf_agents
        # for this carrier, sample the number of contacts and distribute them 
        xinf = x.inf # get infection status to get the right beta 
        if xinf == CARC
            beta = p.beta[1]
        elseif xinf == CARW
            beta = p.beta[2]
        elseif xinf == CARY
            beta = p.beta[3]
        end
        cpview = @view cp[:, x.ag]
        num_contacts = 0.7*rand(cm[x.ag]) # 30% of contacts are not type of contacts where disease will transmit
        totalcontacts += num_contacts
        
        @inbounds for i in 1:len_agbrac # for each agegroup 
            agc = round(Int64, cpview[i]*num_contacts) # get the number of contacts in that age group
            @inbounds for _ in 1:agc # loop through each contacts in that age group
                idx = 1+rand(Int32)&(length(buckets[i]) - 1)  # fast way to get a random index from 1 to size of bucket
                sid = buckets[i][idx]
                s = humans[sid]
                inf_will_happen = false
                if s.inf == SUS && s.c_inf < p.cap_value
                    inf_will_happen = rand() < beta * (1 - s.eff)
                end
                if s.inf == REC && s.p_inf ≠ xinf && s.c_inf < p.cap_value
                    inf_will_happen = rand() < 0.8*beta*(1 - s.eff)
                end
                if inf_will_happen
                    s.swap = xinf
                    incidence[i] += 1 # increase incidence in that age group
                end
            end
        end
        # @info "Agent " x
        # @info "Number of contacts: $num_contacts"
        # @info "Distributed contacts: $distr_contacts"
        # @info "Sampled contacts: $(println(sampled_contacts))"
        # @info "Incidence count: $incidence_cnt"
        # @info "Initial count of inf: $(length(inf_agents))"
    end
    return incidence
end

# this is used in a column wise manner 
# proportion of contacts
const cp = [0.014343162 0.011575898 0.004675017 0.002821965 0.004472203 0.007066033 0.003255765 0.004821674
0.091842802 0.150787443 0.079932172 0.027541389 0.023681223 0.042415078 0.021608503 0.028818523
0.09939079 0.214187497 0.484787881 0.11870858 0.031478152 0.066029 0.04049985 0.044553958
0.051119748 0.06288288 0.101147741 0.415251709 0.105505516 0.052618437 0.040878276 0.039276555
0.11070661 0.073886603 0.036652053 0.14417507 0.32830656 0.123901986 0.106253193 0.075337033
0.469601298 0.355290395 0.206407471 0.193043224 0.332644147 0.464768414 0.401075992 0.277292104
0.103366324 0.086468964 0.060480654 0.071644288 0.136274933 0.191601559 0.307158154 0.203504845
0.059629266 0.044920321 0.025917012 0.026813775 0.037637266 0.051599494 0.079270267 0.326395308]

# number of contacts
const cm = [NegativeBinomial(30.87010127, 0.349008742), NegativeBinomial(15.46544683, 0.187678531), NegativeBinomial(18.45686473, 0.145058802), NegativeBinomial(18.25392743, 0.129283893), NegativeBinomial(6.084348835, 0.044927381), NegativeBinomial(5.422478002, 0.04699169), NegativeBinomial(3.987398168, 0.042517453), NegativeBinomial(7.030203244, 0.146547036)]
