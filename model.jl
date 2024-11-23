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
    prob_of_imd::Float64 = 0.0  # NOT USED
    adj_vax_opt::String = "r1"  # r1 (Option 1 - only one dose, older age group), r2 (Option 2 - first dose shifted to 15 year olds)
    adj_vax_cov::Float64 = 0.61 # vaccine coverage for cov2 
    adj_vax_age::Int64 = 16     # age group for cov2 
    cov1_age::Int64 = 11        # age for cov1 
    cap_value::Int16 = 99       # maximum number of infections
end
# constant variables
const humans = Array{Human}(undef, 0) 
const p = ModelParameters()  ## setup default parameters
const POPSIZE = 100000

# distributions 
const DISTR_CARRIAGE = Poisson(42) # From Feldman paper, 9.6 months ~ 42 weeks
const DISTR_RECOVER = Poisson(245)
const VAC_EFF = Beta(5.68408748, 2.145234118)
const VAC_DURATION = Poisson(261) # seyed to go check

includet("helpers.jl")

### Main simulation functions
function run_model(simid, startyear, endyear, save_sim = false)
    # main time loop function 
    @info "Running simulation ID: $simid"
    @info "Model parameters" dump(p)
    @info "Saving simulation data? $(save_sim)"
    
    # set seed to simulation id
    Random.seed!(simid*264)

    tm_years = startyear:endyear 
    ytm = repeat(tm_years, inner=52) # create an array with the years for indexing the year
    totaltime = length(ytm) # in weeks for the time loop 
    @info "Year Range: $(tm_years), Run time (in years:) $(length(tm_years)), (in weeks:) $totaltime" 
    
    # data collection -- allocate memory 
    ss_inc = zeros(Int16, length(tm_years), length(AG_BRAC), 3, 2) # time(in years) x ag x 3carriage x 2vaccine
    ss_carriage = zeros(Int64, 6, totaltime) # data 3 rows for C, W, Y since

    @inbounds for t in 1:totaltime 
        currentyear = ytm[t]
        if currentyear >= 2005 # vaccine starts in 2005
            if t % 52 == 0 
                cov1, cov2 = get_coverage(currentyear) 
                init_vaccine(cov1, cov2, currentyear)
            end 
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
    if save_sim
        @info "saving simulation to jld2"
        file = jldopen("/data/imd_abm_calibration/sim$(simid)n.jld2", "w") 
        file["h"] = humans #save the humans object 
        close(file)
    end
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

    @info "disease proportions" get_disease_prop()
    
    return 
end

function init_disease()
    @info "Initializing disease..."

    # check if disease is already in the population (perhaps through jldread)
    if findfirst(x -> x.inf ∈ (CARC, CARW, CARY), humans) ≠ nothing
        @info "Disease already initialized in population"
        return
    end

    ag_carriage_prev = rand.([Uniform(0.028, 0.085), Uniform(0.031, 0.091), 
                                Uniform(0.042, 0.118), Uniform(0.055, 0.174), 
                                Uniform(0.106, 0.334), Uniform(0.053, 0.264), 
                                Uniform(0.034, 0.125), Uniform(0.026, 0.090)])
    #ag_carriage_prev = [0.085, 0.091, 0.118, 0.174, 0.334, 0.264, 0.125, 0.090] ./ 4 # equilibrium values

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
    # helper function to get the vaccine coverage value for a particular year
    # only works for years between 2005 and 2025 (+ fixed values for 2026 - 2035), otherwise returns 0
    cov1 = 0.0 
    cov2 = 0.0 
    
    # create coverage values from 2005 to 2024 (fixed vector of 20 elements)
    fdc = [0.13, 0.19, 0.35, 0.45, 0.55, 0.62, 0.67, 0.69, 0.73, 0.74, 0.75, 0.77, 0.79, 0.80, 0.81, 0.83, 0.86, 0.9, 0.9, 0.9]
    ddc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.06, 0.10, 0.14, 0.18, 0.23, 0.29, 0.33, 0.37, 0.41, 0.46, 0.49, 0.55, 0.61, 0.61, 0.61]
    @assert length(fdc) == length(ddc) == 20 

    # calculate coverage values for each year past 2024
    if year ∈ 2005:2024
        cov1 = fdc[year - 2004]
        cov2 = ddc[year - 2004]
    end 
    if year > 2024 
        # baseline coverage remains fixed for beyond 2024 
        cov1 = 0.9
        cov2 = 0.61 

        # any adjustments needed due to vaccine scenarios        
        # R1 -- first dose is shifted to older age (adjusted by adj_vax_age) 
        # Do this by setting cov1 = 0.0 so anyone selected under cov2 gets first dose
        if p.adj_vax_opt == "r1"     
            cov1 = 0.0 
            cov2 = p.adj_vax_cov
        end

        # R2 -- first dose is shifted to 15 year olds (only for 2029+, see logic by Seyed)
        # Do this by setting cov1 = 0.0 (but making sure cov2 only selects those who got first dose)
        # Then in 2029+, set cov1 value
        if p.adj_vax_opt == "r2"
            #p.adj_vax_cov = 90
            if year == 2025
                cov1 = 0.0
                cov2 = (p.adj_vax_cov > 0.83) ? 0.83 : p.adj_vax_cov 
                # since in 2020, the first dose coverage (for 11 year olds) is 0.83 
            elseif year == 2026
                cov1 = 0.0
                cov2 = (p.adj_vax_cov > 0.86) ? 0.86 : p.adj_vax_cov 
                # since in 2021, the first dose coverage (for 11 year olds) is 0.86
            elseif year ∈ 2027:2028
                cov1 = 0.0
                cov2 = p.adj_vax_cov
            elseif year == 2029
                cov1 = 0.90
                cov2 = p.adj_vax_cov # (i.e., this would be 61% (or 90%) of those who got vaccine at age 11 in 2024)
                # since it doesn't make sense that second dose coverage for new cohort would be 90% but old cohort would be 61%. 
                # if coverage goes to 90% for 15 year olds, it should be 90% for 11 year olds as well
            elseif year ==  2030
                cov1 = 0.90
                cov2 = 0.0 # because this would try to select 11 year olds from 2025 but we set cov1 to zero in 2025 
            else # 2031 onward
                cov1 = 0.90
                cov2 = p.adj_vax_cov
            end
        end

        # R3 -- 100% coverage for those who got their first dose
        # This is very similar to baseline -- just that second dose coverage is increased to include everyone 
        if p.adj_vax_opt == "r3"    
            cov1 = 0.9
            cov2 = 0.9 # to ensure all of the first dose gets second dose, this is modified in the vaccine function
        end

    end
    @info "coveragə values scenario: $(p.adj_vax_opt) year $year cov1 $cov1 cov2 $cov2"
    return cov1, cov2
end

function init_vaccine(cov1, cov2, year)
    # implementing vaccine every year starting in 2005
    # cov1, cov2 are coverage values for first dose and second dose
    cov1 + cov2 == 0 && return

    # find everyone 11 years of age (or if r2, change the age to 15) 
    # if r1, cov1 == 0 (past 2025) anyways so age doesn't matter -- this is NOT error checked
    cov1_agegroup = (572:623) ## baseline 11 years of age
    if p.adj_vax_opt == "r2" 
        if year >= 2029
            cov1_agegroup = convert_year_to_wkrange(15)
        end
    end 
    # (cov1 > 0 && p.adj_vax_opt == "r1") && error("R1 vaccine should have cov1=0")
    cc1 = shuffle!(findall(x -> x.age ∈ cov1_agegroup, humans))
    tv1 = round(Int64, cov1 * length(cc1)) # get total that will get vaccinated
    e1 = @view cc1[1:tv1] # view their indices
    for idx in e1  # vaccinate them
        x = humans[idx]
        vaccinate(x)
    end

    # for cov2 (could be first or second dose), 
    # find *everyone* 16 (or 15, 17) years of age and then multiply by coverage value
    cov2_agegroup = convert_year_to_wkrange(16) # baseline scenario
    if p.adj_vax_opt == "r1"
        cov2_agegroup = convert_year_to_wkrange(p.adj_vax_age)    
    end
    if p.adj_vax_opt == "r2" 
        if year >= 2031 # 2031 and beyond, 17 year olds get second dose (since 2029 is when 15 year olds get their first dose)
            cov2_agegroup = convert_year_to_wkrange(17)
        end
    end
    if p.adj_vax_opt == "r3"
        if year >= 2025 
            cov2_agegroup = convert_year_to_wkrange(p.adj_vax_age)   
        end
    end

    # find the total number of people in this age group to vaccinate 
    # the cov2 value is "overall" coverage (i.e. not x% of those who got first dose, just x% of the population)
    # but we make sure that in baseline (and other scenarios), those who got first dose get second dose with cov2 coverage
    cc2 = shuffle!(findall(x -> x.age ∈ cov2_agegroup && x.vac < 2, humans))
    tv2 = round(Int64, cov2 * length(cc2)) - 1 # total number of agents vaccinated, substract 1 for stability

    # find agents who are eligble for second dose (i.e. got their first dose)
    # these would be the first agents to get the second dose -- and then select the rest from cc2 (it would be their first dose)
    cc2_vax = shuffle!(findall(x -> x.age ∈ cov2_agegroup && x.vac == 1, humans))
    
    if p.adj_vax_opt == "r3"
        if year >= 2025
            tv2 = length(cc2_vax) # force everyone who got first dose to get second dose regardess or cov2 value
        end
    end

    @info "vaccine funct before error (scen: $(p.adj_vax_opt)) dynamics year: $year, cov2: $cov2, cc2=$(length(cc2)), cc2_vax=$(length(cc2_vax)), tv2=$tv2"

    # error checks
    # in baseline and r2 scenarios, we want cov2 to be only for second dose 
    # so make sure the length of cc2_vax is greater than tv2
    # for r1, cc2_vax is likely going to become zero as we stopped first dose (so folks under cov2 get their first dose eventually)

    if p.adj_vax_opt ∈ ("baseline", "r2", "r3")
        if length(cc2_vax) < tv2
            @info "Not enough agents in cc2_vax to vaccinate for second dose"
            @info "total needed to vaccinate: $tv2, total with one dose $(length(cc2_vax)), year: $year"
            flush(stdout)
            error("Not enough agents in cc2_vax to vaccinate for second dose, year: $year")
        end
    end

    # select the agents for cov2 -- prioritize those needing second dose
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
    if x.eff == 0.0 # if efficacy is zero (i.e. first dose, or first dose has expired)
        x.eff = rand(VAC_EFF) # set efficacy
    end
    x.exp = x.age + rand(VAC_DURATION) # set expiry
    x.vac == 3 && error("ID $(x.idx) / 3 doses - vac: $(x.vac), age: $(convert_week_to_year(x.age))")
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
    # x.vac == 0 -- don't remove agents that are vaccinated for stability of vaccine functions
    eligible = findall(x -> x.age > 52 && x.age < 5200 && x.vac == 0, humans)
    
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
        age_in_years =  convert_week_to_year(x.age) + 1 # +1 since minimum age is 0 (which corresponds to first element)
        cpview = @view cp2[age_in_years, :]
        num_contacts = rand(cm2[age_in_years])
        @inbounds for i in 1:len_agbrac # for each agegroup 
            println("\t age: $age_in_years, i: $i, num_contacts: $num_contacts, cpview[$i]: $(cpview[i])")
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

# new number of contacts - based on close contacts that could transmit disease
const cm2 = [NegativeBinomial(1.708484963, 0.178043502), NegativeBinomial(1.67406475, 0.178752528), NegativeBinomial(1.684319952, 0.182453156), NegativeBinomial(3.935115597, 0.213825286), NegativeBinomial(5.62372754, 0.226051986), NegativeBinomial(8.36429731, 0.244435748), NegativeBinomial(9.399752712, 0.247576571), NegativeBinomial(9.133376626, 0.246888209), NegativeBinomial(8.47384027, 0.243291142), NegativeBinomial(7.652973511, 0.234610047), NegativeBinomial(6.786706354, 0.226065252), NegativeBinomial(12.06374946, 0.28810984), NegativeBinomial(16.50662559, 0.318978315), NegativeBinomial(15.61536228, 0.314598388), NegativeBinomial(12.13371878, 0.294463291), NegativeBinomial(11.47371482, 0.282821747), NegativeBinomial(13.24406355, 0.308208167), NegativeBinomial(9.876544477, 0.284916941), NegativeBinomial(5.402783889, 0.267246467), NegativeBinomial(4.295320118, 0.300979887), NegativeBinomial(3.754867959, 0.280956928), NegativeBinomial(3.298001329, 0.261899924), NegativeBinomial(2.298682545, 0.247628611), NegativeBinomial(2.118567863, 0.249264748), NegativeBinomial(1.842076765, 0.238259862), NegativeBinomial(1.861074307, 0.235591721), NegativeBinomial(2.062849607, 0.246753833), NegativeBinomial(1.887117492, 0.218910001), NegativeBinomial(1.993278702, 0.225408063), NegativeBinomial(2.062592611, 0.229240823), NegativeBinomial(1.750975239, 0.200082296), NegativeBinomial(1.596041941, 0.19674729), NegativeBinomial(1.508137469, 0.195975586), NegativeBinomial(1.392575726, 0.185383222), NegativeBinomial(1.219652709, 0.161516606), NegativeBinomial(1.224219901, 0.169648364), NegativeBinomial(1.164190081, 0.15789289), NegativeBinomial(1.189352628, 0.159399609), NegativeBinomial(1.243485598, 0.162875323), NegativeBinomial(1.280669405, 0.163191049), NegativeBinomial(1.220903313, 0.157515519), NegativeBinomial(1.298465064, 0.163181869), NegativeBinomial(1.24586315, 0.15097413), NegativeBinomial(1.285678918, 0.155959364), NegativeBinomial(1.416864579, 0.161794901), NegativeBinomial(1.465084307, 0.161337184), NegativeBinomial(1.571989066, 0.165057473), NegativeBinomial(1.676772101, 0.165951775), NegativeBinomial(1.731074387, 0.163398706), NegativeBinomial(1.825436006, 0.166959042), NegativeBinomial(1.893305784, 0.16585532), NegativeBinomial(1.864898545, 0.163540272), NegativeBinomial(1.881659622, 0.166114882), NegativeBinomial(1.875925716, 0.1619058), NegativeBinomial(1.953407505, 0.167270896), NegativeBinomial(1.934956454, 0.172570498), NegativeBinomial(1.818844319, 0.162087916), NegativeBinomial(1.797133057, 0.164257751), NegativeBinomial(1.758091232, 0.160199431), NegativeBinomial(1.722502948, 0.159367018), NegativeBinomial(1.659508321, 0.1567125), NegativeBinomial(1.748541152, 0.163718918), NegativeBinomial(1.789470554, 0.167808426), NegativeBinomial(1.800372716, 0.170902974), NegativeBinomial(1.746392406, 0.168600113), NegativeBinomial(1.773849817, 0.188052249), NegativeBinomial(1.802683628, 0.189810445), NegativeBinomial(1.919210901, 0.199007531), NegativeBinomial(1.992347626, 0.203102906), NegativeBinomial(2.074133087, 0.211408225), NegativeBinomial(2.119487638, 0.214414799), NegativeBinomial(2.070734269, 0.214569759), NegativeBinomial(2.006668149, 0.213510129), NegativeBinomial(1.95275151, 0.208434641), NegativeBinomial(2.034086827, 0.223156056), NegativeBinomial(1.964359318, 0.21486585), NegativeBinomial(1.941783544, 0.215118326), NegativeBinomial(1.975647402, 0.225210516), NegativeBinomial(1.866528153, 0.216594186), NegativeBinomial(1.827753485, 0.219598676), NegativeBinomial(1.725773728, 0.220953516), NegativeBinomial(1.84560635, 0.220663201), NegativeBinomial(1.858925579, 0.223904201), NegativeBinomial(1.743000325, 0.211329113), NegativeBinomial(1.449837055, 0.197829045), NegativeBinomial(1.587465762, 0.210783395), NegativeBinomial(1.590328094, 0.20470894), NegativeBinomial(1.731684565, 0.211055323), NegativeBinomial(1.788503799, 0.208934826), NegativeBinomial(1.870113624, 0.213480521), NegativeBinomial(2.059604075, 0.225090781), NegativeBinomial(2.494395002, 0.24294805), NegativeBinomial(2.790368274, 0.253065516), NegativeBinomial(2.998355429, 0.259152368), NegativeBinomial(3.294377229, 0.272195607), NegativeBinomial(3.61595014, 0.281527252), NegativeBinomial(3.98196963, 0.294155172), NegativeBinomial(4.326808075, 0.308812697), NegativeBinomial(4.645313413, 0.317614946), NegativeBinomial(4.570904842, 0.310068333), NegativeBinomial(4.875158823, 0.323600578), NegativeBinomial(1.048067941, 0.344547483)]
const cp2 = [0.041571327 0.170399902 0.092439067 0.047217941 0.103464268 0.483679784 0.058756076 0.002471634
0.047753559 0.219213666 0.093323403 0.046058315 0.093331671 0.444005372 0.053951298 0.002362716
0.033976818 0.265184772 0.103968511 0.04574695 0.08489932 0.413695363 0.050228875 0.002299391
0.010680918 0.416170203 0.250327561 0.039926218 0.039681079 0.212979076 0.029000301 0.001234645
0.005465927 0.407582547 0.347006679 0.041616345 0.026544727 0.149137805 0.021745042 0.000900929
0.003137957 0.121938146 0.700336612 0.041146102 0.016499069 0.101121542 0.01517129 0.000649281
0.002255659 0.078475065 0.770631121 0.039276145 0.013481456 0.082540199 0.012785795 0.000554561
0.001824533 0.043947795 0.809492783 0.043848157 0.012602407 0.075588647 0.012147626 0.000548053
0.001523396 0.029104402 0.815128946 0.060104604 0.012328049 0.069534098 0.011727261 0.000549245
0.001358774 0.023751262 0.762357439 0.121394475 0.012449781 0.066216815 0.011895923 0.000575531
0.001247051 0.020128222 0.62225552 0.265258165 0.013344217 0.064585067 0.012556852 0.000624906
0.000850235 0.009923498 0.249064465 0.672604567 0.01155568 0.045843386 0.009647091 0.000511078
0.000639564 0.00560439 0.062202392 0.873205312 0.014320022 0.035342291 0.008224113 0.000461916
0.000567918 0.004841533 0.029090216 0.904793736 0.020335116 0.03172961 0.008145098 0.000496773
0.000501672 0.00337097 0.012631485 0.791813295 0.156163141 0.027277793 0.007714588 0.000527057
0.000297863 0.001800036 0.005144843 0.541238082 0.430495109 0.015608015 0.005011263 0.00040479
0.000117568 0.000711123 0.001756767 0.368519038 0.619408049 0.006852812 0.002426507 0.000208134
4.89855E-05 0.000290101 0.000591222 0.203189211 0.792198095 0.002598293 0.000983758 0.000100334
4.42522E-05 0.000238725 0.000384047 0.049200108 0.946338818 0.002833079 0.000860596 0.000100374
5.7328E-05 0.000281166 0.00035469 0.005484499 0.987713835 0.0049306 0.001035891 0.000141991
7.59731E-05 0.000359082 0.000361197 0.000868574 0.984088643 0.01258475 0.001445852 0.000215929
0.000126374 0.000596256 0.000538469 0.000559586 0.952871482 0.042740063 0.002244221 0.000323548
0.000278724 0.001368763 0.001168137 0.000922104 0.844005193 0.146965176 0.004651909 0.000639994
0.000487725 0.002404282 0.002004926 0.00140525 0.623613893 0.361515354 0.007560322 0.001008247
0.00078035 0.00387323 0.003255838 0.002004986 0.398720884 0.578820589 0.011106427 0.001437696
0.001073406 0.005397889 0.004789454 0.00271699 0.164585177 0.804460815 0.015129897 0.001846373
0.001367888 0.006835101 0.00636527 0.003296728 0.059992004 0.902477407 0.017512747 0.002152855
0.00166263 0.008372305 0.0081789 0.003965221 0.027665627 0.928066103 0.019653092 0.002436122
0.00194874 0.010040185 0.01018769 0.004884013 0.020993605 0.927617314 0.021622251 0.002706202
0.002258614 0.011648201 0.012291865 0.005981901 0.020800442 0.920332153 0.023696636 0.002990187
0.002772814 0.014456754 0.015698981 0.007743551 0.01881847 0.91010619 0.026968679 0.003434561
0.003266331 0.017185124 0.019510164 0.009860198 0.021410436 0.894168075 0.030644238 0.003955433
0.00354984 0.019389385 0.022706664 0.011897201 0.023911081 0.880751313 0.033391174 0.004403344
0.003780078 0.021324466 0.026212589 0.014369602 0.026204509 0.867359699 0.035929158 0.004819899
0.003904602 0.022684264 0.029171491 0.016567504 0.028921519 0.854879579 0.038621715 0.005249327
0.004010315 0.023961065 0.032474021 0.019052055 0.029058427 0.844102263 0.04164384 0.005698013
0.003965597 0.024284029 0.034656477 0.020995277 0.031652752 0.83437671 0.044016737 0.006052422
0.003767684 0.024089685 0.037258433 0.023308772 0.03421138 0.8250601 0.045891444 0.006412501
0.003625619 0.023563546 0.038298862 0.024927002 0.036827283 0.81838001 0.047563845 0.006813832
0.003417199 0.022472341 0.038948621 0.026597729 0.038660158 0.814046028 0.048827499 0.007030425
0.003430002 0.022413822 0.041259528 0.029578985 0.040592837 0.803076502 0.052074602 0.007573722
0.003094042 0.020574183 0.039404735 0.030277926 0.04231738 0.799070363 0.057090823 0.008170546
0.002846506 0.019336125 0.038081661 0.03077845 0.045233987 0.791107279 0.0637205 0.008895492
0.002570786 0.017325155 0.035651633 0.030843385 0.047285916 0.787532256 0.069262929 0.00952794
0.002411201 0.015979292 0.033039586 0.030232528 0.049039744 0.782806836 0.076395067 0.010095745
0.0022621 0.014680443 0.030337583 0.028901782 0.050254801 0.778362001 0.084520799 0.010680491
0.002078588 0.013576802 0.027752922 0.027669821 0.051352962 0.768836062 0.097614245 0.011118599
0.002027378 0.012831353 0.025444879 0.026367493 0.052392944 0.739026369 0.130139248 0.011770336
0.001923345 0.012004674 0.023172273 0.024748407 0.052524876 0.668147609 0.205253512 0.012225304
0.00183926 0.011526798 0.02120784 0.023316074 0.052616126 0.541174214 0.335602688 0.012717
0.001817173 0.011226018 0.020333481 0.022142621 0.052416859 0.425410125 0.453372411 0.013281313
0.001795696 0.011066443 0.019440014 0.021471751 0.053112993 0.308029462 0.571049955 0.014033687
0.001829579 0.011052568 0.018983309 0.02065634 0.053771515 0.249687578 0.628924873 0.015094236
0.001807553 0.01109898 0.018900447 0.020137636 0.053958557 0.229946621 0.64812326 0.016026946
0.001824242 0.011249374 0.019359015 0.0196821 0.053753646 0.225932763 0.650938336 0.017260523
0.001915813 0.011605622 0.019538871 0.019528833 0.053750236 0.224270536 0.650163987 0.019226101
0.001884585 0.011588761 0.019845414 0.019170755 0.053054857 0.222281454 0.651423066 0.02075111
0.001928716 0.011862619 0.020653829 0.019118384 0.052910598 0.225143281 0.645291338 0.023091234
0.001926883 0.012254223 0.021646341 0.019373105 0.052583797 0.226046818 0.640335771 0.025833061
0.001928879 0.012589191 0.021982529 0.019306789 0.051581682 0.223731492 0.639905239 0.0289742
0.001971061 0.012996721 0.023152557 0.019903928 0.049995331 0.223273892 0.63399435 0.03471216
0.001788754 0.011795371 0.020768225 0.017820441 0.046581525 0.219607466 0.632386995 0.049251223
0.001619823 0.010615603 0.018452893 0.015544717 0.042982056 0.213916194 0.621784666 0.075084047
0.00148442 0.00950517 0.01629357 0.013801107 0.04144998 0.211744806 0.573635256 0.132085691
0.001316187 0.008425679 0.014196845 0.012065205 0.039866732 0.205711323 0.477974255 0.240443772
0.001285494 0.00799308 0.013008044 0.011038383 0.038921376 0.173866592 0.367664411 0.386222621
0.001123982 0.006891618 0.01117303 0.009749147 0.037517028 0.166107937 0.245684252 0.521753007
0.000982501 0.0059 0.009348087 0.008795121 0.035693727 0.159796411 0.181187015 0.598297138
0.000834103 0.00506395 0.007948224 0.008264757 0.03497458 0.154982309 0.148814926 0.63911715
0.000720157 0.004285426 0.006720539 0.007873016 0.03348173 0.149394617 0.128507169 0.669017348
0.000607987 0.003623007 0.005635174 0.007871555 0.032604518 0.143916267 0.113242477 0.692499014
0.000534782 0.003186433 0.005083873 0.007923438 0.032871725 0.14386507 0.105914574 0.700620105
0.000472384 0.002821852 0.004642504 0.008049921 0.033318799 0.14454973 0.100282883 0.705861927
0.000402055 0.002392668 0.004282998 0.008046987 0.033542637 0.141772648 0.094233395 0.715326612
0.000342748 0.002049221 0.004187973 0.008309118 0.033472974 0.140742872 0.090306164 0.72058893
0.000293969 0.001763788 0.004096817 0.008138215 0.034168749 0.138822846 0.087030131 0.725685486
0.000251191 0.001443974 0.003784325 0.007659632 0.032006062 0.136163902 0.083855866 0.734835048
0.000211922 0.001294177 0.003785898 0.007779471 0.03232697 0.13384565 0.080628514 0.740127398
0.000186313 0.001216209 0.003894444 0.007950045 0.032933852 0.134119101 0.079933763 0.739766273
0.000162639 0.00122298 0.003981406 0.008187829 0.034200478 0.135252628 0.080692442 0.736299598
0.000143403 0.001270587 0.004151887 0.008484151 0.034502121 0.13744163 0.082907445 0.731098776
0.000165409 0.001420108 0.004655146 0.009378126 0.03532341 0.136312372 0.089629527 0.723115901
0.000192199 0.001657347 0.005357029 0.010597181 0.035877639 0.139911977 0.097347372 0.709059256
0.000228664 0.001969824 0.006359081 0.01268437 0.03749193 0.146879611 0.110147255 0.684239265
0.000307258 0.00264824 0.008474579 0.016703381 0.043704277 0.169314492 0.139845068 0.619002704
0.000268182 0.002823801 0.009314219 0.017710321 0.041187232 0.168458807 0.145608834 0.614628604
0.000234556 0.002808359 0.010337175 0.018838393 0.038376565 0.168551568 0.148204137 0.612649247
0.000205278 0.002475211 0.011720924 0.019233202 0.035973057 0.16944584 0.149321544 0.611624944
0.000179818 0.002162844 0.01301877 0.019115228 0.03392427 0.171110235 0.150119647 0.610369187
0.000157773 0.001892329 0.014118712 0.018646914 0.032295465 0.173813777 0.150598453 0.608476579
0.000138821 0.001659698 0.014898266 0.017862484 0.030946151 0.178372008 0.150708131 0.605414442
0.000116052 0.001382446 0.014007113 0.016280899 0.027837401 0.175374689 0.142054763 0.622946636
9.92158E-05 0.001177036 0.012097426 0.015523954 0.02550211 0.176769849 0.136770212 0.632060197
8.59041E-05 0.0010144 0.009427565 0.015310277 0.023686887 0.180461607 0.133553028 0.636460333
7.47924E-05 0.000878593 0.00722684 0.014505224 0.022560454 0.184558095 0.131219828 0.638976173
6.53052E-05 0.000762674 0.005491026 0.013198221 0.021556693 0.189532361 0.129719722 0.639673998
5.72153E-05 0.000663846 0.00414697 0.011625832 0.020692799 0.195929187 0.128726594 0.638157556
5.02849E-05 0.000549195 0.003146674 0.009707143 0.020162824 0.203802354 0.126686036 0.635895489
4.43957E-05 0.000440646 0.002402815 0.007462984 0.02026977 0.212990165 0.125261227 0.631127997
3.95259E-05 0.000347613 0.001853915 0.005757861 0.02027517 0.223766597 0.12503146 0.622927858
3.57171E-05 0.000272837 0.001455046 0.00451885 0.020536967 0.238490564 0.12657236 0.608117659]