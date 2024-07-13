using StatsBase, Random
using DelimitedFiles
using Distributions
using DelimitedFiles

@enum INFTYPE UNDEF=0 SUS=1 CARC=2 CARW=3 CARY=4 IMD=5 REC=6

# define an agent and all agent properties
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
end
Base.show(io::IO, ::MIME"text/plain", z::Human) = dump(z)

## system parameters
Base.@kwdef mutable struct ModelParameters
    maxtime::Int64 = 520 # in weeks 
    beta::Vector{Float64} = [0.5, 0.5, 0.5] # for Carriage 
    prob_of_imd::Float64 = 0.001
end
# constant variables
const humans = Array{Human}(undef, 0) 
const p = ModelParameters()  ## setup default parameters
const POPSIZE = 100000

# distributions 
const DISTR_CARRIAGE = Gamma(99.31, 0.131) # convert to weeks 
const DISTR_RECOVER = Poisson(244)

include("helpers.jl")

### Main simulate function 
function simulate(simid) 
    # print simulation details 
    @info "Running simid: $simid on processor ID: $(myid())"
    
    #init_state()
    init_agents() # reset agents
    init_disease()

    # data collection
    res_carriage = zeros(Int64, 3, p.maxtime) # 3 rows for C, W, Y since      
    for t in 1:p.maxtime
        timestep()
        _c, _w, _y = collect_data()
        #res_carriage[1:3, t] .= (_c, _w, _y)
        res_carriage[1:3, t] .= (_c, _w, _y)
    end
    return res_carriage
end

function collect_data()
    # keep optimized since this runs in the time loop
    _c, _w, _y = 0, 0, 0  
    distr = countmap(Int.([x.inf for x in humans]))
    haskey(distr, Int(CARC)) && (_c = distr[2])
    haskey(distr, Int(CARW)) && (_w = distr[3])
    haskey(distr, Int(CARY)) && (_y = distr[4])    
    return (_c, _w, _y)
end

### Iniltialization Functions
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

function init_agents() 
    # non-equilibrium age distribution
    # pop_distr = [1102, 1077, 1118, 1137, 1163, 1187, 1218, 1229, 1225, 1221, 1223, 1236, 1262, 1289, 1338, 1335, 1312, 1300, 1308, 1296, 1289, 1312, 1313, 1292, 1290, 1293, 1302, 1325, 1353, 1374, 1405, 1426, 1430, 1389, 1365, 1349, 1352, 1350, 1322, 1341, 1338, 1318, 1311, 1257, 1229, 1213, 1171, 1183, 1160, 1175, 1231, 1304, 1283, 1234, 1207, 1210, 1228, 1274, 1304, 1303, 1306, 1302, 1277, 1252, 1239, 1214, 1159, 1130, 1088, 1040, 998, 948, 901, 873, 850, 891, 625, 595, 573, 573, 484, 427, 385, 351, 321, 279, 251, 224, 187, 167, 145, 125, 105, 84, 68, 53, 40, 30, 21, 15, 23]

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
    @debug "Initializing disease..."
    ag_carriage_prev = rand.([Uniform(0.028, 0.085), Uniform(0.031, 0.091), 
                                Uniform(0.042, 0.118), Uniform(0.055, 0.174), 
                                Uniform(0.106, 0.334), Uniform(0.053, 0.264), 
                                Uniform(0.034, 0.125), Uniform(0.026, 0.090)])

    _sero_prop = [0.0, 0.0, 0.0] # Index: 1=C, 2=W, 3=Y -- ORDER MATTERS
    _sero_prop[1] = rand(Uniform(0.41, 0.47)) # SERO C
    _sero_prop[2] = rand(Uniform(0.02, 0.05)) # SERO ּW    
    _sero_prop[3] = 1 - _sero_prop[1] - _sero_prop[2] # SERO Y

    sero_pop = Categorical(_sero_prop)
    @debug "...sero props $_sero_prop"

    # error check -- make sure population is clean
    findfirst(x -> (x.inf ≠ SUS || x.swap ≠ UNDEF), humans) ≠ nothing && error("Population not clean")

    for ag in 1:length(AG_BRAC)
        prev = ag_carriage_prev[ag]
        elig = findall(x -> x.ag == ag && rand() < prev, humans)
        for id in elig 
            x = humans[id]
            x.swap = INFTYPE(rand(sero_pop) + 1)  # since CARC = 2, CARY = 3, CARW = 4
            activate_swaps(x)
        end
    end
    @debug "...Disease Initialized" get_disease_prop()
end

function timestep() 
    # main timestep function -- advanced simulation by 1 week 
    # age dynamics - increase age by one and then run age dynamics
    for x in humans
        x.age += 1 
        x.ag = findfirst(Base.Fix1(∈, x.age), AG_BRAC)
        x.tis += 1 
        swaps = naturalhistory(x) # move through the natural history of the disease first         
        activate_swaps(x) # deal with the swap dynamics either through naturalhistory OR transmission
    end
    (inc, wc) = transmission() # naturalhistory affects swaps of carriage and rec. Swap for carriage is irrelevant, swap might rec -> sus, but activate_swaps comes after, it doesn matter
    age_dynamics() 
    return 
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
    if x.swap ≠ UNDEF 
        if x.swap in (CARC, CARW, CARY)
            move_to_carriage(x)
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
end

@inline function move_to_carriage(x::Human) 
    x.inf = x.swap # so will be CARC, CARY, CARW
    x.c_inf += 1 # raise infection cnt
    x.st = round(Int16, 4*rand(DISTR_CARRIAGE)) 
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

    # # turn them into newborns! Use infoid() to print statistics
    newborn.(deathids)
    return 
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
    return
end

function transmission() 
    incidence_cnt = 0 
    wasted_contact = 0 
    inf_agents = humans[findall(x -> x.inf ∈ (CARC, CARW, CARY), humans)]
    length(inf_agents) == 0 && return (incidence_cnt, wasted_contact) # no carriers to transmit
    buckets = [findall(x -> x.ag == i, humans) for i = 1:length(AG_BRAC)]
    shuffle!.(buckets)
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
        
        @inbounds for i in 1:length(AG_BRAC)
            agc = round(Int64, cpview[i]*num_contacts)
            @inbounds for _ in 1:agc # go through each count 
                idx = 1+rand(Int32)&(length(buckets[i]) - 1)  # fast way to get a random index from 1 to size of bucket
                sid = buckets[i][idx]
                s = humans[sid]
                inf_will_happen = false
                if s.inf == SUS
                    inf_will_happen = rand() < beta
                end
                if s.inf == REC && s.p_inf ≠ xinf 
                    inf_will_happen = rand() < 0.8*beta
                end
                if inf_will_happen
                    s.swap = xinf
                    incidence_cnt += 1
                else
                    wasted_contact += 1
                end
            end
            
            #sampled_contacts = buckets[i][rand(1:(length(buckets[i])), agc)]
            #sampled_contacts = @view buckets[i][1:agc] # this is fast but not what I want
            #sampled_contacts = sample(buckets[i], agc, replace=false)
            #sampled_contacts = rand(buckets[i], agc)
            # for sid in sampled_contacts 
            #     s = humans[sid]
            #     inf_will_happen = false
            #     if s.inf == SUS
            #         inf_will_happen = rand() < beta
            #     end
            #     if s.inf == REC && s.p_inf ≠ xinf 
            #         inf_will_happen = rand() < 0.8*beta
            #     end
            #     if inf_will_happen
            #         s.swap = xinf
            #         incidence_cnt += 1
            #     else
            #         wasted_contact += 1
            #     end
            # end
        end
        #distr_contacts = round.(Int64, cmview .* num_contacts) # will give the number of contacts for each age group
        #_sampled_contacts = sample.(buckets, distr_contacts, replace=false) # broadcast sample over 8 buckets and contacts for 8 age group
        #sampled_contacts = Base.Iterators.flatten(_sampled_contacts)

        # @info "Agent " x
        # @info "Number of contacts: $num_contacts"
        # @info "Distributed contacts: $distr_contacts"
        # @info "Sampled contacts: $(println(sampled_contacts))"
        # @info "Incidence count: $incidence_cnt"
        # @info "Initial count of inf: $(length(inf_agents))"
    end
    return (incidence_cnt, wasted_contact)
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
