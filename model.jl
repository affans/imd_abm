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
function run_model(simid, startyear, endyear; save_sim = false)
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
        file = jldopen("/data/imd_abm_calibration_nov26/sim$(simid)n.jld2", "w") 
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
    filename = "/data/imd_abm_calibration_nov26/sim$(id)n.jld2"
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
    @info "coverage values scenario: $(p.adj_vax_opt) year $year cov1 $cov1 cov2 $cov2"
    return cov1, cov2
end

function init_vaccine(cov1, cov2, year)
    # implementing vaccine every year starting in 2005
    # cov1, cov2 are coverage values for first dose and second dose
    cov1 + cov2 == 0.0 && return

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
    @info "init_vaccine: cov1 dyn (scen: $(p.adj_vax_opt)), year: $year, cov1: $cov1 cov1_agegroup: $cov1_agegroup, cc1=$(length(cc1)), tv1=$tv1"

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
    for x in inf_agents
        # for this carrier, sample the number of contacts and distribute them 
        xinf = x.inf # get infection status to get the right beta 
        if xinf == CARC
            beta = p.beta[1]
        elseif xinf == CARW
            beta = p.beta[2]
        elseif xinf == CARY
            beta = p.beta[3]
        end
        # nov 25: the new contact matrix is age-specific with maximum year of 101
        # however x.age could be 5252 (making year 102) which is a bounds issue 
        # so we substract x.age - 1 to get minimum 5251 which is year 101 
        # for those on the 'edges' this may make them one year younger than they are 
        # but over simulations/averages, this is not a big deal
        age_in_years = convert_week_to_year(x.age - 1) + 1
        cpview = vec(cp2[age_in_years, :])
        num_contacts = rand(cm2[age_in_years])
        for i in 1:len_agbrac # for each agegroup
            agc = round(Int64, cpview[i]*num_contacts) # get the number of contacts in that age group
            for _ in 1:agc # loop through each contacts in that age group
                idx = 1+rand(Int32)&(length(buckets[i]) - 1)  # fast way to get a random index from 1 to size of bucket
                sid = buckets[i][idx]
                s = humans[sid]
                inf_will_happen = false
                if s.inf == SUS && s.c_inf < p.cap_value
                    inf_will_happen = rand() < beta * (1 - s.eff)
                end
                if s.inf == REC && s.p_inf ≠ xinf && s.c_inf < p.cap_value
                    inf_will_happen = rand() < 0.8 * beta * (1 - s.eff)
                end
                if inf_will_happen
                    s.swap = xinf
                    incidence[i] += 1 # increase incidence in that age group
                end
            end
        end
    end
    return incidence
end

const cm2 = [NegativeBinomial(1.708484963, 0.178043502), NegativeBinomial(1.67406475, 0.178752528), NegativeBinomial(1.684319952, 0.182453156), NegativeBinomial(3.935115597, 0.213825286), NegativeBinomial(5.62372754, 0.226051986), NegativeBinomial(8.36429731, 0.244435748), NegativeBinomial(9.399752712, 0.247576571), NegativeBinomial(9.133376626, 0.246888209), NegativeBinomial(8.47384027, 0.243291142), NegativeBinomial(7.652973511, 0.234610047), NegativeBinomial(6.786706354, 0.226065252), NegativeBinomial(12.06374946, 0.28810984), NegativeBinomial(16.50662559, 0.318978315), NegativeBinomial(15.61536228, 0.314598388), NegativeBinomial(12.13371878, 0.294463291), NegativeBinomial(11.47371482, 0.282821747), NegativeBinomial(13.24406355, 0.308208167), NegativeBinomial(9.876544477, 0.284916941), NegativeBinomial(5.402783889, 0.267246467), NegativeBinomial(4.295320118, 0.300979887), NegativeBinomial(3.754867959, 0.280956928), NegativeBinomial(3.298001329, 0.261899924), NegativeBinomial(2.298682545, 0.247628611), NegativeBinomial(2.118567863, 0.249264748), NegativeBinomial(1.842076765, 0.238259862), NegativeBinomial(1.861074307, 0.235591721), NegativeBinomial(2.062849607, 0.246753833), NegativeBinomial(1.887117492, 0.218910001), NegativeBinomial(1.993278702, 0.225408063), NegativeBinomial(2.062592611, 0.229240823), NegativeBinomial(1.750975239, 0.200082296), NegativeBinomial(1.596041941, 0.19674729), NegativeBinomial(1.508137469, 0.195975586), NegativeBinomial(1.392575726, 0.185383222), NegativeBinomial(1.219652709, 0.161516606), NegativeBinomial(1.224219901, 0.169648364), NegativeBinomial(1.164190081, 0.15789289), NegativeBinomial(1.189352628, 0.159399609), NegativeBinomial(1.243485598, 0.162875323), NegativeBinomial(1.280669405, 0.163191049), NegativeBinomial(1.220903313, 0.157515519), NegativeBinomial(1.298465064, 0.163181869), NegativeBinomial(1.24586315, 0.15097413), NegativeBinomial(1.285678918, 0.155959364), NegativeBinomial(1.416864579, 0.161794901), NegativeBinomial(1.465084307, 0.161337184), NegativeBinomial(1.571989066, 0.165057473), NegativeBinomial(1.676772101, 0.165951775), NegativeBinomial(1.731074387, 0.163398706), NegativeBinomial(1.825436006, 0.166959042), NegativeBinomial(1.893305784, 0.16585532), NegativeBinomial(1.864898545, 0.163540272), NegativeBinomial(1.881659622, 0.166114882), NegativeBinomial(1.875925716, 0.1619058), NegativeBinomial(1.953407505, 0.167270896), NegativeBinomial(1.934956454, 0.172570498), NegativeBinomial(1.818844319, 0.162087916), NegativeBinomial(1.797133057, 0.164257751), NegativeBinomial(1.758091232, 0.160199431), NegativeBinomial(1.722502948, 0.159367018), NegativeBinomial(1.659508321, 0.1567125), NegativeBinomial(1.748541152, 0.163718918), NegativeBinomial(1.789470554, 0.167808426), NegativeBinomial(1.800372716, 0.170902974), NegativeBinomial(1.746392406, 0.168600113), NegativeBinomial(1.773849817, 0.188052249), NegativeBinomial(1.802683628, 0.189810445), NegativeBinomial(1.919210901, 0.199007531), NegativeBinomial(1.992347626, 0.203102906), NegativeBinomial(2.074133087, 0.211408225), NegativeBinomial(2.119487638, 0.214414799), NegativeBinomial(2.070734269, 0.214569759), NegativeBinomial(2.006668149, 0.213510129), NegativeBinomial(1.95275151, 0.208434641), NegativeBinomial(2.034086827, 0.223156056), NegativeBinomial(1.964359318, 0.21486585), NegativeBinomial(1.941783544, 0.215118326), NegativeBinomial(1.975647402, 0.225210516), NegativeBinomial(1.866528153, 0.216594186), NegativeBinomial(1.827753485, 0.219598676), NegativeBinomial(1.725773728, 0.220953516), NegativeBinomial(1.84560635, 0.220663201), NegativeBinomial(1.858925579, 0.223904201), NegativeBinomial(1.743000325, 0.211329113), NegativeBinomial(1.449837055, 0.197829045), NegativeBinomial(1.587465762, 0.210783395), NegativeBinomial(1.590328094, 0.20470894), NegativeBinomial(1.731684565, 0.211055323), NegativeBinomial(1.788503799, 0.208934826), NegativeBinomial(1.870113624, 0.213480521), NegativeBinomial(2.059604075, 0.225090781), NegativeBinomial(2.494395002, 0.24294805), NegativeBinomial(2.790368274, 0.253065516), NegativeBinomial(2.998355429, 0.259152368), NegativeBinomial(3.294377229, 0.272195607), NegativeBinomial(3.61595014, 0.281527252), NegativeBinomial(3.98196963, 0.294155172), NegativeBinomial(4.326808075, 0.308812697), NegativeBinomial(4.645313413, 0.317614946), NegativeBinomial(4.570904842, 0.310068333), NegativeBinomial(4.875158823, 0.323600578), NegativeBinomial(1.048067941, 0.344547483)]
const cp2 = [0.042 0.17 0.092 0.047 0.103 0.484 0.059 0.002
    0.048 0.219 0.093 0.046 0.093 0.444 0.054 0.002
    0.034 0.265 0.104 0.046 0.085 0.414 0.05 0.002
    0.011 0.416 0.25 0.04 0.04 0.213 0.029 0.001
    0.005 0.408 0.347 0.042 0.027 0.149 0.022 0.001
    0.003 0.122 0.7 0.041 0.016 0.101 0.015 0.001
    0.002 0.078 0.771 0.039 0.013 0.083 0.013 0.001
    0.002 0.044 0.809 0.044 0.013 0.076 0.012 0.001
    0.002 0.029 0.815 0.06 0.012 0.07 0.012 0.001
    0.001 0.024 0.762 0.121 0.012 0.066 0.012 0.001
    0.001 0.02 0.622 0.265 0.013 0.065 0.013 0.001
    0.001 0.01 0.249 0.673 0.012 0.046 0.01 0.001
    0.001 0.006 0.062 0.873 0.014 0.035 0.008 0
    0.001 0.005 0.029 0.905 0.02 0.032 0.008 0
    0.001 0.003 0.013 0.792 0.156 0.027 0.008 0.001
    0 0.002 0.005 0.541 0.43 0.016 0.005 0
    0 0.001 0.002 0.369 0.619 0.007 0.002 0
    0 0 0.001 0.203 0.792 0.003 0.001 0
    0 0 0 0.049 0.946 0.003 0.001 0
    0 0 0 0.005 0.988 0.005 0.001 0
    0 0 0 0.001 0.984 0.013 0.001 0
    0 0.001 0.001 0.001 0.953 0.043 0.002 0
    0 0.001 0.001 0.001 0.844 0.147 0.005 0.001
    0 0.002 0.002 0.001 0.624 0.362 0.008 0.001
    0.001 0.004 0.003 0.002 0.399 0.579 0.011 0.001
    0.001 0.005 0.005 0.003 0.165 0.804 0.015 0.002
    0.001 0.007 0.006 0.003 0.06 0.902 0.018 0.002
    0.002 0.008 0.008 0.004 0.028 0.928 0.02 0.002
    0.002 0.01 0.01 0.005 0.021 0.928 0.022 0.003
    0.002 0.012 0.012 0.006 0.021 0.92 0.024 0.003
    0.003 0.014 0.016 0.008 0.019 0.91 0.027 0.003
    0.003 0.017 0.02 0.01 0.021 0.894 0.031 0.004
    0.004 0.019 0.023 0.012 0.024 0.881 0.033 0.004
    0.004 0.021 0.026 0.014 0.026 0.867 0.036 0.005
    0.004 0.023 0.029 0.017 0.029 0.855 0.039 0.005
    0.004 0.024 0.032 0.019 0.029 0.844 0.042 0.006
    0.004 0.024 0.035 0.021 0.032 0.834 0.044 0.006
    0.004 0.024 0.037 0.023 0.034 0.825 0.046 0.006
    0.004 0.024 0.038 0.025 0.037 0.818 0.048 0.007
    0.003 0.022 0.039 0.027 0.039 0.814 0.049 0.007
    0.003 0.022 0.041 0.03 0.041 0.803 0.052 0.008
    0.003 0.021 0.039 0.03 0.042 0.799 0.057 0.008
    0.003 0.019 0.038 0.031 0.045 0.791 0.064 0.009
    0.003 0.017 0.036 0.031 0.047 0.788 0.069 0.01
    0.002 0.016 0.033 0.03 0.049 0.783 0.076 0.01
    0.002 0.015 0.03 0.029 0.05 0.778 0.085 0.011
    0.002 0.014 0.028 0.028 0.051 0.769 0.098 0.011
    0.002 0.013 0.025 0.026 0.052 0.739 0.13 0.012
    0.002 0.012 0.023 0.025 0.053 0.668 0.205 0.012
    0.002 0.012 0.021 0.023 0.053 0.541 0.336 0.013
    0.002 0.011 0.02 0.022 0.052 0.425 0.453 0.013
    0.002 0.011 0.019 0.021 0.053 0.308 0.571 0.014
    0.002 0.011 0.019 0.021 0.054 0.25 0.629 0.015
    0.002 0.011 0.019 0.02 0.054 0.23 0.648 0.016
    0.002 0.011 0.019 0.02 0.054 0.226 0.651 0.017
    0.002 0.012 0.02 0.02 0.054 0.224 0.65 0.019
    0.002 0.012 0.02 0.019 0.053 0.222 0.651 0.021
    0.002 0.012 0.021 0.019 0.053 0.225 0.645 0.023
    0.002 0.012 0.022 0.019 0.053 0.226 0.64 0.026
    0.002 0.013 0.022 0.019 0.052 0.224 0.64 0.029
    0.002 0.013 0.023 0.02 0.05 0.223 0.634 0.035
    0.002 0.012 0.021 0.018 0.047 0.22 0.632 0.049
    0.002 0.011 0.018 0.016 0.043 0.214 0.622 0.075
    0.001 0.01 0.016 0.014 0.041 0.212 0.574 0.132
    0.001 0.008 0.014 0.012 0.04 0.206 0.478 0.24
    0.001 0.008 0.013 0.011 0.039 0.174 0.368 0.386
    0.001 0.007 0.011 0.01 0.038 0.166 0.246 0.522
    0.001 0.006 0.009 0.009 0.036 0.16 0.181 0.598
    0.001 0.005 0.008 0.008 0.035 0.155 0.149 0.639
    0.001 0.004 0.007 0.008 0.033 0.149 0.129 0.669
    0.001 0.004 0.006 0.008 0.033 0.144 0.113 0.692
    0.001 0.003 0.005 0.008 0.033 0.144 0.106 0.701
    0 0.003 0.005 0.008 0.033 0.145 0.1 0.706
    0 0.002 0.004 0.008 0.034 0.142 0.094 0.715
    0 0.002 0.004 0.008 0.033 0.141 0.09 0.721
    0 0.002 0.004 0.008 0.034 0.139 0.087 0.726
    0 0.001 0.004 0.008 0.032 0.136 0.084 0.735
    0 0.001 0.004 0.008 0.032 0.134 0.081 0.74
    0 0.001 0.004 0.008 0.033 0.134 0.08 0.74
    0 0.001 0.004 0.008 0.034 0.135 0.081 0.736
    0 0.001 0.004 0.008 0.035 0.137 0.083 0.731
    0 0.001 0.005 0.009 0.035 0.136 0.09 0.723
    0 0.002 0.005 0.011 0.036 0.14 0.097 0.709
    0 0.002 0.006 0.013 0.037 0.147 0.11 0.684
    0 0.003 0.008 0.017 0.044 0.169 0.14 0.619
    0 0.003 0.009 0.018 0.041 0.168 0.146 0.615
    0 0.003 0.01 0.019 0.038 0.169 0.148 0.613
    0 0.002 0.012 0.019 0.036 0.169 0.149 0.612
    0 0.002 0.013 0.019 0.034 0.171 0.15 0.61
    0 0.002 0.014 0.019 0.032 0.174 0.151 0.608
    0 0.002 0.015 0.018 0.031 0.178 0.151 0.605
    0 0.001 0.014 0.016 0.028 0.175 0.142 0.623
    0 0.001 0.012 0.016 0.026 0.177 0.137 0.632
    0 0.001 0.009 0.015 0.024 0.18 0.134 0.636
    0 0.001 0.007 0.015 0.023 0.185 0.131 0.639
    0 0.001 0.005 0.013 0.022 0.19 0.13 0.64
    0 0.001 0.004 0.012 0.021 0.196 0.129 0.638
    0 0.001 0.003 0.01 0.02 0.204 0.127 0.636
    0 0 0.002 0.007 0.02 0.213 0.125 0.631
    0 0 0.002 0.006 0.02 0.224 0.125 0.623
    0 0 0.001 0.005 0.021 0.238 0.127 0.608]
    
