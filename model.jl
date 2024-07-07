using StatsBase, Random
using DelimitedFiles

@enum INFTYPE SUS1=0 CAR1=1 IMD1=2 REC1=3 SUS2=4 CAR2=5 IMD2=6 REC2=7

# define an agent and all agent properties
Base.@kwdef mutable struct Human
    idx::Int64 = 0 
    age::Int64 = 0 # age in weeks
end
Base.show(io::IO, ::MIME"text/plain", z::Human) = dump(z)

## system parameters
Base.@kwdef mutable struct ModelParameters
    popsize::Int64 = 100000
    numofsims::Int64 = 100
    maxtime::Int64 = 520 # in weeks 
end
# constant variables
const humans = Array{Human}(undef, 0) 
const p = ModelParameters()  ## setup default parameters

include("helpers.jl")

### Iniltialization Functions 
init_state() = init_state(ModelParameters())
function init_state(ip::ModelParameters)
    # the p is a global const
    # the ip is an incoming different instance of parameters 
    # copy the values from ip to p. 
    ip.popsize == 0 && error("no population size given")
    for x in propertynames(p)
        setfield!(p, x, getfield(ip, x))
    end
    # resize the human array to change population size
    resize!(humans, p.popsize)

    # initialize with the human object 
    for i in 1:p.popsize
        humans[i] = Human()
        x = humans[i]
        x.idx = i
    end
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
    @assert sum(pop_distr) == p.popsize
    @assert length(sampled_ages) == p.popsize

    # if rounding issues, try this (from the rsv canada project) 
    # push!(sf_agegroups, fill(6, abs(p.popsize - length(sf_agegroups)))...) # if length(sf_agegroups) != p.popsize # because of rounding issues
    for i in 1:p.popsize
        x = humans[i]
        x.age = sampled_ages[i]
    end

    return get_age_distribution()
end

function timestep() 
    # main timestep function -- advanced simulation by 1 week 

    # age dynamics - increase age by one and then run age dynamics
    for x in humans
        x.age += 1 
    end
    age_dynamics()
    return 
end

function age_dynamics() 
    # function deals with newborns and deaths

    # get the number of agents switch from <1 to 1-4
    num_to_die = length(findall(x -> x.age == 52, humans)) # only length because we need to select this many random
    # find all individuals that will die due to natural death
    deathids = findall(x -> x.age == 5252, humans)

    #println("total number of newborns to introduce: $num_to_die")
    #println("total number of natural deaths: $(length(deathids))")

    # need to a total of `num_to_die` including the natural deaths 
    # so fill in the rest of deaths by randomly sampling from the population
    # ensure that those transitioning to 1-4 and 100+ are not part of sampling
    eligible = findall(x -> x.age > 52 && x.age < 5200, humans) 

    # add to the death IDs randomly selected ones 
    for _ in 1:(num_to_die - length(deathids))
        push!(deathids, rand(eligible))
    end

    # turn them into newborns!
    newborn.(deathids)
    #infoid.(deathids) # print the statistics on the death ids
end

@inline function newborn(id) 
    # converts a human to a newborn
    x = humans[id]
    x.age = 0    
    return true # return status -- can be used for counting
end
