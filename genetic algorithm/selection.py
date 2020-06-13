import sys
import os
import numpy as np
import random

from onepointcrossover import onepointcrossover
from mutation import mutation

from nsga2 import nondominatedsort, crowdingdistance, crowdedcomparison
from lastfrontselection import lastfrontselection
from uftournselection import uftournselection

# Generation number
generation = int(sys.argv[1])

# Number of individuals in a population
population = int(sys.argv[2])

# Number of scenarios of same individuals
scenarios = int(sys.argv[3])

# Total number of simulations
n_simulations = population*scenarios

# Number of guides
n_guides = int(sys.argv[4])

# Batch size of a single array job
batch_size = int(sys.argv[5])

# Number of partitions of the simulations of a single generation
parts = int(sys.argv[6])

# Number of simulations in a single partition
simulations_part = int(sys.argv[7])

# SLURM JOBID for the simulations of current generation
jobid = []
jobid.append(str(sys.argv[8]))

# If there are several partitions, load all SLURM JOBIDs
if parts >= 2:
    jobid.append(str(sys.argv[9]))
    if parts >= 3:
        jobid.append(str(sys.argv[10]))
        if parts >= 4:
            jobid.append(str(sys.argv[11]))

# Minimization/maximization of objectives
# We are both minimizing expected evacuation time and variance of evacuation time
str1 = "min"
str2 = "min"

# Probabilities for scenarios
p_scenario = [0.3, 0.2, 0.2, 0.3]
# Check if length of p_scenario matches number of scenarios
if len(p_scenario) != scenarios:
    raise Exception('Wrong size probability vector!')
# Check if p_scenario really contains probabilities
if np.sum(p_scenario) != 1:
    raise Exception('Not a probability vector!')
for i in range(len(p_scenario)):
    if p_scenario[i] < 0 or p_scenario[i] > 1:
        raise Exception('Not a probability vector!')

# Probability for crossover
CXPB = 0.85
CXPB_tags = 0.85

# Probability for mutation
MUTPB = 0.1
MUTPB_tags = 0.1

# Write simulation results to a single file
all_positions = []
all_evactimes = []

# For optimization of the run time of the genetic algorithm in SLURM, the array jobs might be partitioned and have
# different batch sizes.
for i in range(0, parts):
    for j in range(0, int(simulations_part)):
        fname = "{}{}{}{}{}".format('slurm-', jobid[i], '_', j, '.out')
        with open(fname) as infile:
            lines = [line.rstrip('\n') for line in infile]
            # If the number of lines equal 2 times the batch size, everything should be fine.
            # All odd rows are positions of the guides, and even rows are the evacuation times.
            if len(lines) == 2*batch_size:
                for k in range(batch_size):
                    all_positions.append(lines[0+2*k])
                    all_evactimes.append(float(lines[1+2*k]))
            else:
                raise Exception("{}{}".format('Something wrong with simulation result', fname))
            #    for k in range(len(lines):
            #        if (k % 2) == 0:
            #        else:
                #all_positions.append(0)
                #all_output_first.append(-10)


# There might be a few invalid runs (because of faults in the spawning).
# Replace the results of invalid runs with 0.0
#all_evactimes = []
#for i in range(0, n_simulations):
#    try:
#        all_evactimes.append(float(all_output_first[i]))
#    except ValueError:
#        all_evactimes.append(-10)

# The output data is in the format, where there are all the scenarios of the first individual, then of the second...
# So, if there is a invalid output, we need to replace all scenarios of that individual.
waste = np.argwhere(np.asarray(all_evactimes)==-10)
invalid_individuals = []

for i in range(len(waste)):

    # Check what is the scenario number for the invalid output.
    invalid_scenario_number = i % scenarios
    # Calculate the individual number
    individual_number = i - i % scenarios

    # Check that the scenario of a individual has not yet been added to the list of invalid outputs.
    already_saved = 0
    for k in range(len(invalid_individuals)):
        if invalid_individuals[k] == individual_number:
            already_saved = 1

    if already_saved == 1:
        continue

    invalid_individuals.append(individual_number)

n_invalid_individuals = len(invalid_individuals)

# If there are invalid outputs, replace them with a random valid individual
if n_invalid_individuals > 0:

    # The indices of the valid individuals
    all_individuals = [i for i in range(population)]
    valid_individuals = [i for i in all_individuals if i not in invalid_individuals]
    indxs_replacement = []
    for i in range(0, n_invalid_individuals):
        rand_indx = np.random.randint(0, population-n_invalid_individuals, 1)[0]
        indxs_replacement.append(valid_individuals[rand_indx])
        all_evactimes[invalid_individuals[i] * scenarios:(invalid_individuals[i] + 1) * scenarios] = all_evactimes[indxs_replacement[i] * scenarios:(indxs_replacement[i] + 1) * scenarios]
        all_positions[invalid_individuals[i] * scenarios:(invalid_individuals[i] + 1) * scenarios] = all_positions[indxs_replacement[i] * scenarios:(indxs_replacement[i] + 1) * scenarios]

# Calculate expected evacuation times and CVaR of evacuation times for different individuals in the population.
expected_evactimes = np.zeros(population)
cvar_evactimes = np.zeros(population)

# Expected total evacuation time
for i in range(0, population):
    for j in range(0, scenarios):
        expected_evactimes[i] += p_scenario[j] * all_evactimes[i * scenarios + j]

# CVaR
quantile = 0.05 # (actually this is 1-quantile)
for i in range(0, population):
    individual_evactimes = all_evactimes[i*scenarios:(i+1)*scenarios]
    sorted_index = sorted(range(scenarios), key=lambda j: individual_evactimes[j])
    sorted_index = sorted_index[::-1]
    sorted_individual_evactimes = [individual_evactimes[j] for j in sorted_index]
    sorted_p_scenario = [p_scenario[j] for j in sorted_index]

    # Cumulative probabilities
    cum_p_scenario = [np.sum(sorted_p_scenario[0:j+1]) for j in range(scenarios)]

    # Check what total evacuation times can occur at the q-quantile, and calculate their conditional probabilities.
    upper = min(j for j in cum_p_scenario if j >= quantile)
    tail_p_scenario = [sorted_p_scenario[j] for j in range(scenarios) if cum_p_scenario[j] <= upper]
    n_tail_scenarios = len(tail_p_scenario)
    tail_evactimes  = [sorted_individual_evactimes[j] for j in range(scenarios) if cum_p_scenario[j] <= upper]

    scenario_weight = tail_p_scenario

    if upper != quantile:
        if n_tail_scenarios == 1:
            last_weight = quantile
        else:
            last_weight = quantile - cum_p_scenario[n_tail_scenarios-2]
        scenario_weight[-1] = last_weight

    scenario_weight = [scenario_weight[j]/sum(scenario_weight) for j in range(n_tail_scenarios)]

    scenario_weight = np.asarray(scenario_weight)
    tail_evactimes = np.asarray(tail_evactimes)

    # Calculate the conditional expected evacuation times at the q-quantile
    cvar_evactimes[i] = np.sum(tail_evactimes*scenario_weight)

expected_evactimes.tolist()
cvar_evactimes.tolist()

# Save data
# Save results of all the runs (whole invalid scenarios have been replaced)
np.savetxt("{}{}{}".format('evactimes_', generation, '.txt'), np.asarray(all_evactimes), fmt='%.14f')

# Save the positions of all the runs (whole invalid scenarios have been replaced)
f = open("{}{}{}".format('positions_', generation, '.txt'), "w")
for i in range(n_simulations):
    f.write("{}{}".format(all_positions[i], '\n'))
f.close()

# Save the penaltyscores of this population
np.savetxt("{}{}{}".format('expected_evactimes_', generation, '.txt'), np.asarray(expected_evactimes), fmt='%.14f')
np.savetxt("{}{}{}".format('cvar_evactimes_', generation, '.txt'), np.asarray(cvar_evactimes), fmt='%.14f')


# Load data of individuals in the current generation,
exits1 = np.loadtxt("{}{}{}".format('exits1_', generation, '.txt'), dtype=int)
exits1 = exits1[0::scenarios]
cells1 = np.loadtxt("{}{}{}".format('cells1_', generation, '.txt'), dtype=int)
cells1 = cells1[0::scenarios]

if n_guides >= 2:
    exits2 = np.loadtxt("{}{}{}".format('exits2_', generation, '.txt'), dtype=int)
    cells2 = np.loadtxt("{}{}{}".format('cells2_', generation, '.txt'), dtype=int)
    exits2 = exits2[0::scenarios]
    cells2 = cells2[0::scenarios]

    if n_guides >= 3:
        exits3 = np.loadtxt("{}{}{}".format('exits3_', generation, '.txt'), dtype=int)
        cells3 = np.loadtxt("{}{}{}".format('cells3_', generation, '.txt'), dtype=int)
        exits3 = exits3[0::scenarios]
        cells3 = cells3[0::scenarios]

        if n_guides >= 4:
            exits4 = np.loadtxt("{}{}{}".format('exits4_', generation, '.txt'), dtype=int)
            cells4 = np.loadtxt("{}{}{}".format('cells4_', generation, '.txt'), dtype=int)
            exits4 = exits4[0::scenarios]
            cells4 = cells4[0::scenarios]

            if n_guides >= 5:
                exits5 = np.loadtxt("{}{}{}".format('exits5_', generation, '.txt'), dtype=int)
                cells5 = np.loadtxt("{}{}{}".format('cells5_', generation, '.txt'), dtype=int)
                exits5 = exits5[0::scenarios]
                cells5 = cells5[0::scenarios]

                if n_guides >= 6:
                    exits6 = np.loadtxt("{}{}{}".format('exits6_', generation, '.txt'), dtype=int)
                    cells6 = np.loadtxt("{}{}{}".format('cells6_', generation, '.txt'), dtype=int)
                    exits6 = exits6[0::scenarios]
                    cells6 = cells6[0::scenarios]

                    if n_guides >= 7:
                        exits7 = np.loadtxt("{}{}{}".format('exits7_', generation, '.txt'), dtype=int)
                        cells7 = np.loadtxt("{}{}{}".format('cells7_', generation, '.txt'), dtype=int)
                        exits7 = exits7[0::scenarios]
                        cells7 = cells7[0::scenarios]

                        if n_guides >= 8:
                            exits8 = np.loadtxt("{}{}{}".format('exits8_', generation, '.txt'), dtype=int)
                            cells8 = np.loadtxt("{}{}{}".format('cells8_', generation, '.txt'), dtype=int)
                            exits8 = exits8[0::scenarios]
                            cells8 = cells8[0::scenarios]

                            if n_guides >= 9:
                                exits9 = np.loadtxt("{}{}{}".format('exits9_', generation, '.txt'), dtype=int)
                                cells9 = np.loadtxt("{}{}{}".format('cells9_', generation, '.txt'), dtype=int)
                                exits9 = exits9[0::scenarios]
                                cells9 = cells9[0::scenarios]

                                if n_guides >= 10:
                                    exits10 = np.loadtxt("{}{}{}".format('exits10_', generation, '.txt'), dtype=int)
                                    cells10 = np.loadtxt("{}{}{}".format('cells10_', generation, '.txt'), dtype=int)
                                    exits10 = exits10[0::scenarios]
                                    cells10 = cells10[0::scenarios]

# Use eliticism.
# Take the n_simulations best individuals from the parents and the offspring
# The population of the first generation will always be random.
# Already the second generation will consist of the n_simulations best individuals from the first generation
# and the crossovered and mutated offspring.
if generation > 0:
    
    # Open the results of the previous generation and extend them to the results of current generation

    # Total evacuation times of all the runs
    # Load parent data
    parents_evactimes = np.loadtxt("{}{}{}".format('nondominated_evactimes_', generation - 1, '.txt'))
    # Transform to list
    parents_evactimes = parents_evactimes.tolist()
    # Extend the offspring data to parent data
    parents_evactimes.extend(all_evactimes)
    all_evactimes = parents_evactimes

    # Positions of all the runs
    # Load parent data
    with open("{}{}{}".format('nondominated_positions_', generation - 1, '.txt')) as infile:
        lines2 = [line.rstrip('\n') for line in infile]
    # Extend the offspring data to parent data
    lines2.extend(all_positions)
    all_positions = lines2

    # Expected evacuation times
    # Load parent data
    parents_expected_evactimes = np.loadtxt("{}{}{}".format('nondominated_expected_evactimes_', generation - 1, '.txt'))
    # Transform to list
    parents_expected_evactimes = parents_expected_evactimes.tolist()
    # Extend the offspring data to parent data
    parents_expected_evactimes.extend(expected_evactimes)
    objectives1 = parents_expected_evactimes

    # CVaR of evacuation times
    # Load parent data
    parents_cvar_evactimes = np.loadtxt("{}{}{}".format('nondominated_cvar_evactimes_', generation - 1, '.txt'))
    # Transform to list
    parents_cvar_evactimes = parents_cvar_evactimes.tolist()
    # Extend the offspring data to parent data
    parents_cvar_evactimes.extend(cvar_evactimes)
    objectives2 = parents_cvar_evactimes

else:
    objectives1 = expected_evactimes
    objectives2 = cvar_evactimes

# TODO: Here comes the NSGA-2 procedure
# Combine parent and offspring population
# THIS WAS DONE ALREADY ABOVE

# Perform the fast-non-dominated-sort to attain fronts and ranks of solutions
F, rank = nondominatedsort(objectives1, objectives2, str1, str2)

# Initialize next set of parents
next_parent = []

# Distances
distances = []

# Ranks of parents
next_ranks = []

# Initialize iteration counter
i = 0

# Until the parent population is filled
while len(next_parent) + len(F[i]) <= population:
    
    # Calculate crowding distance in F_i
    F_obj1 = [objectives1[i] for i in F[i]]
    F_obj2 = [objectives2[i] for i in F[i]]
    F_dist = crowdingdistance(F_obj1, F_obj2, str1, str2)
    distances.extend(F_dist)

    # Include the ith nondominated front in the parent population
    next_parent.extend(F[i])

    # Include the ranks of the solutions in the ith nondominated front
    temp_ranks = np.ones(len(F[i]))*(i+1)
    temp_ranks = temp_ranks.tolist()
    next_ranks.extend(temp_ranks)

    # Increase iteration counter
    i += 1

    # If next parents has been filled break out of loop
    if len(next_parent) == population:
        break

# Go through the last front, and add individuals to the next parent using the LASTFRONTSELECTION procedure
if len(next_parent) != population:
    F_obj1 = [objectives1[j] for j in F[i]]
    F_obj2 = [objectives2[j] for j in F[i]]
    F_rank = [float(i+1) for j in F[i]]
    F_dist = crowdingdistance(F_obj1, F_obj2, str1, str2)
    lastfront = lastfrontselection(F_obj1, F_obj2, F_dist, population - len(next_parent))
    # Correct the indices
    temp_lastfront_indxs = []
    temp_lastfront_ranks = []
    temp_lastfront_dists = []
    for j in range(len(lastfront)):
        temp_lastfront_indxs.append(F[i][lastfront[j]])
        temp_lastfront_ranks.append(float(i+1))
        temp_lastfront_dists.append(F_dist[j])
    
   # Include the remaining solutions from i+1th nondominated front.
    distances.extend(temp_lastfront_dists)
    next_ranks.extend(temp_lastfront_ranks)
    next_parent.extend(temp_lastfront_indxs)

next_objectives1 = []
next_objectives2 = []
for j in range(len(distances)):
    next_objectives1.append(objectives1[next_parent[j]])
    next_objectives2.append(objectives2[next_parent[j]])

# Unique fitness based tournament selection
temp_selected = uftournselection(next_objectives1, next_objectives2, next_ranks, distances) 
selected = []
for j in range(len(distances)):
    selected.append(next_parent[temp_selected[j]])

if generation > 0:
    # Check if some of the chosen individuals where of "invalid" type. Replace their
    # indices with their replacement. => This way we won't breed any of the invalid
    # types.
    # THIS IS DONE SO THAT WE USE THE CORRECT EXIT AND CELL DATA WHEN BREEDING!
    if n_invalid_individuals > 0:
        for i in range(0, n_invalid_individuals):
            wrongly_chosen = np.argwhere(np.asarray(selected)==(invalid_individuals[i] + population)) #shouldn't here be a " + population"
            if len(wrongly_chosen) > 0:
                for j in range(0, len(wrongly_chosen)):
                    selected[wrongly_chosen[j][0]] = indxs_replacement[i] + population #shouldn't here be a " + population"

else:
    # Check if some of the chosen individuals where of "invalid" type. Replace their
    # indices with their replacement. => This way we won't breed any of the invalid
    # types.
    # THIS IS DONE SO THAT WE USE THE CORRECT EXIT AND CELL DATA WHEN BREEDING!
    if n_invalid_individuals > 0:
        for i in range(0, n_invalid_individuals):
            wrongly_chosen = np.argwhere(np.asarray(selected)==invalid_individuals[i])
            if len(wrongly_chosen) > 0:
                for j in range(0, len(wrongly_chosen)):
                    selected[wrongly_chosen[j][0]] = indxs_replacement[i]


# Load data of individuals selected for breeding
if generation > 0:
    exits1_prev = np.loadtxt("{}{}{}".format('nondominated_exits1_', generation-1, '.txt'), dtype=int)
    cells1_prev = np.loadtxt("{}{}{}".format('nondominated_cells1_', generation-1, '.txt'), dtype=int)
    exits1 = np.concatenate((exits1_prev, exits1), axis=None)
    cells1 = np.concatenate((cells1_prev, cells1), axis=None)

exits1.tolist()
cells1.tolist()
selected_exits1 = []
selected_cells1 = []
new_exits1 = []
new_cells1 = []

if n_guides >= 2:
    if generation > 0:
        exits2_prev = np.loadtxt("{}{}{}".format('nondominated_exits2_', generation-1, '.txt'), dtype=int)
        cells2_prev = np.loadtxt("{}{}{}".format('nondominated_cells2_', generation-1, '.txt'), dtype=int)
        exits2 = np.concatenate((exits2_prev, exits2), axis=None)
        cells2 = np.concatenate((cells2_prev, cells2), axis=None)

    exits2.tolist()
    cells2.tolist()
    selected_exits2 = []
    selected_cells2 = []
    new_exits2 = []
    new_cells2 = []

    if n_guides >= 3:
        if generation > 0:
            exits3_prev = np.loadtxt("{}{}{}".format('nondominated_exits3_', generation-1, '.txt'), dtype=int)
            cells3_prev = np.loadtxt("{}{}{}".format('nondominated_cells3_', generation-1, '.txt'), dtype=int)
            exits3 = np.concatenate((exits3_prev, exits3), axis=None)
            cells3 = np.concatenate((cells3_prev, cells3), axis=None)

        exits3.tolist()
        cells3.tolist()
        selected_exits3 = []
        selected_cells3 = []
        new_exits3 = []
        new_cells3 = []

        if n_guides >= 4:
            if generation > 0:
                exits4_prev = np.loadtxt("{}{}{}".format('nondominated_exits4_', generation-1, '.txt'), dtype=int)
                cells4_prev = np.loadtxt("{}{}{}".format('nondominated_cells4_', generation-1, '.txt'), dtype=int)
                exits4 = np.concatenate((exits4_prev, exits4), axis=None)
                cells4 = np.concatenate((cells4_prev, cells4), axis=None)

            exits4.tolist()
            cells4.tolist()
            selected_exits4 = []
            selected_cells4 = []
            new_exits4 = []
            new_cells4 = []

            if n_guides >= 5:
                if generation > 0:
                    exits5_prev = np.loadtxt("{}{}{}".format('nondominated_exits5_', generation-1, '.txt'), dtype=int)
                    cells5_prev = np.loadtxt("{}{}{}".format('nondominated_cells5_', generation-1, '.txt'), dtype=int)
                    exits5 = np.concatenate((exits5_prev, exits5), axis=None)
                    cells5 = np.concatenate((cells5_prev, cells5), axis=None)

                exits5.tolist()
                cells5.tolist()
                selected_exits5 = []
                selected_cells5 = []
                new_exits5 = []
                new_cells5 = []

                if n_guides >= 6:
                    if generation > 0:
                        exits6_prev = np.loadtxt("{}{}{}".format('nondominated_exits6_', generation-1, '.txt'), dtype=int)
                        cells6_prev = np.loadtxt("{}{}{}".format('nondominated_cells6_', generation-1, '.txt'), dtype=int)
                        exits6 = np.concatenate((exits6_prev, exits6), axis=None)
                        cells6 = np.concatenate((cells6_prev, cells6), axis=None)

                    exits6.tolist()
                    cells6.tolist()
                    selected_exits6 = []
                    selected_cells6 = []
                    new_exits6 = []
                    new_cells6 = []

                    if n_guides >= 7:
                        if generation > 0:
                            exits7_prev = np.loadtxt("{}{}{}".format('nondominated_exits7_', generation-1, '.txt'), dtype=int)
                            cells7_prev = np.loadtxt("{}{}{}".format('nondominated_cells7_', generation-1, '.txt'), dtype=int)
                            exits7 = np.concatenate((exits7_prev, exits7), axis=None)
                            cells7 = np.concatenate((cells7_prev, cells7), axis=None)

                        exits7.tolist()
                        cells7.tolist()
                        selected_exits7 = []
                        selected_cells7 = []
                        new_exits7 = []
                        new_cells7 = []

                        if n_guides >= 8:
                            if generation > 0:
                                exits8_prev = np.loadtxt("{}{}{}".format('nondominated_exits8_', generation-1, '.txt'), dtype=int)
                                cells8_prev = np.loadtxt("{}{}{}".format('nondominated_cells8_', generation-1, '.txt'), dtype=int)
                                exits8 = np.concatenate((exits8_prev, exits8), axis=None)
                                cells8 = np.concatenate((cells8_prev, cells8), axis=None)

                            exits8.tolist()
                            cells8.tolist()
                            selected_exits8 = []
                            selected_cells8 = []
                            new_exits8 = []
                            new_cells8 = []

                            if n_guides >= 9:
                                if generation > 0:
                                    exits9_prev = np.loadtxt("{}{}{}".format('nondominated_exits9_', generation-1, '.txt'), dtype=int)
                                    cells9_prev = np.loadtxt("{}{}{}".format('nondominated_cells9_', generation-1, '.txt'), dtype=int)
                                    exits9 = np.concatenate((exits9_prev, exits9), axis=None)
                                    cells9 = np.concatenate((cells9_prev, cells9), axis=None)

                                exits9.tolist()
                                cells9.tolist()
                                selected_exits9 = []
                                selected_cells9 = []
                                new_exits9 = []
                                new_cells9 = []

                                if n_guides >= 10:
                                    if generation > 0:
                                        exits10_prev = np.loadtxt("{}{}{}".format('nondominated_exits10_', generation-1, '.txt'), dtype=int)
                                        cells10_prev = np.loadtxt("{}{}{}".format('nondominated_cells10_', generation-1, '.txt'), dtype=int)
                                        exits10 = np.concatenate((exits10_prev, exits10), axis=None)
                                        cells10 = np.concatenate((cells10_prev, cells10), axis=None)

                                    exits10.tolist()
                                    cells10.tolist()
                                    selected_exits10 = []
                                    selected_cells10 = []
                                    new_exits10 = []
                                    new_cells10 = []

# Attain the exits, cells and tags for the individuals selected for breeding
for i in range(population):

    selected_exits1.append(exits1[selected[i]])
    selected_cells1.append(cells1[selected[i]])

    if n_guides >= 2:
        selected_exits2.append(exits2[selected[i]])
        selected_cells2.append(cells2[selected[i]])

        if n_guides >= 3:
            selected_exits3.append(exits3[selected[i]])
            selected_cells3.append(cells3[selected[i]])

            if n_guides >= 4:
                selected_exits4.append(exits4[selected[i]])
                selected_cells4.append(cells4[selected[i]])

                if n_guides >= 5:
                    selected_exits5.append(exits5[selected[i]])
                    selected_cells5.append(cells5[selected[i]])

                    if n_guides >= 6:
                        selected_exits6.append(exits6[selected[i]])
                        selected_cells6.append(cells6[selected[i]])

                        if n_guides >= 7:
                            selected_exits7.append(exits7[selected[i]])
                            selected_cells7.append(cells7[selected[i]])

                            if n_guides >= 8:
                                selected_exits8.append(exits8[selected[i]])
                                selected_cells8.append(cells8[selected[i]])

                                if n_guides >= 9:
                                    selected_exits9.append(exits9[selected[i]])
                                    selected_cells9.append(cells9[selected[i]])

                                    if n_guides >= 10:
                                        selected_exits10.append(exits10[selected[i]])
                                        selected_cells10.append(cells10[selected[i]])

# Start breeding
inds1 = np.arange(0,population,2)
inds2 = np.arange(1,population,2)
for ind1, ind2 in zip(inds1, inds2):
    
    ind1_exits = []
    ind1_cells = []
    ind2_exits = []
    ind2_cells = []

    # Store the data of the individuals in lists
    ind1_exits.append(selected_exits1[ind1])
    ind1_cells.append(selected_cells1[ind1])
    ind2_exits.append(selected_exits1[ind2])
    ind2_cells.append(selected_cells1[ind2])

    if n_guides >= 2:
        ind1_exits.append(selected_exits2[ind1])
        ind1_cells.append(selected_cells2[ind1])
        ind2_exits.append(selected_exits2[ind2])
        ind2_cells.append(selected_cells2[ind2])

        if n_guides >= 3:
            ind1_exits.append(selected_exits3[ind1])
            ind1_cells.append(selected_cells3[ind1])
            ind2_exits.append(selected_exits3[ind2])
            ind2_cells.append(selected_cells3[ind2])

            if n_guides >= 4:
                ind1_exits.append(selected_exits4[ind1])
                ind1_cells.append(selected_cells4[ind1])
                ind2_exits.append(selected_exits4[ind2])
                ind2_cells.append(selected_cells4[ind2])

                if n_guides >= 5:
                    ind1_exits.append(selected_exits5[ind1])
                    ind1_cells.append(selected_cells5[ind1])
                    ind2_exits.append(selected_exits5[ind2])
                    ind2_cells.append(selected_cells5[ind2])

                    if n_guides >= 6:
                        ind1_exits.append(selected_exits6[ind1])
                        ind1_cells.append(selected_cells6[ind1])
                        ind2_exits.append(selected_exits6[ind2])
                        ind2_cells.append(selected_cells6[ind2])

                        if n_guides >= 7:
                            ind1_exits.append(selected_exits7[ind1])
                            ind1_cells.append(selected_cells7[ind1])
                            ind2_exits.append(selected_exits7[ind2])
                            ind2_cells.append(selected_cells7[ind2])

                            if n_guides >= 8:
                                ind1_exits.append(selected_exits8[ind1])
                                ind1_cells.append(selected_cells8[ind1])
                                ind2_exits.append(selected_exits8[ind2])
                                ind2_cells.append(selected_cells8[ind2])

                                if n_guides >= 9:
                                    ind1_exits.append(selected_exits9[ind1])
                                    ind1_cells.append(selected_cells9[ind1])
                                    ind2_exits.append(selected_exits9[ind2])
                                    ind2_cells.append(selected_cells9[ind2])

                                    if n_guides >= 10:
                                        ind1_exits.append(selected_exits10[ind1])
                                        ind1_cells.append(selected_cells10[ind1])
                                        ind2_exits.append(selected_exits10[ind2])
                                        ind2_cells.append(selected_cells10[ind2])
    # Crossover with probability CXPB
    if np.random.uniform(0,1,1) <= CXPB and n_guides >= 2:
        ind1_newexits, ind1_newcells, ind2_newexits, ind2_newcells = onepointcrossover(ind1_exits, ind1_cells, ind2_exits, ind2_cells)

        # Mutate genes of chromosomes
        ind1_newexits, ind1_newcells = mutation(ind1_newexits, ind1_newcells, MUTPB)
        ind2_newexits, ind2_newcells = mutation(ind2_newexits, ind2_newcells, MUTPB)

    else:
        # Mutate genes of chromosomes
        ind1_newexits, ind1_newcells = mutation(ind1_exits, ind1_cells, MUTPB)
        ind2_newexits, ind2_newcells = mutation(ind2_exits, ind2_cells, MUTPB)

    # Append data of offspring to lists
    new_exits1.append(ind1_newexits[0])
    new_exits1.append(ind2_newexits[0])
    new_cells1.append(ind1_newcells[0])
    new_cells1.append(ind2_newcells[0])

    if n_guides >= 2:
        new_exits2.append(ind1_newexits[1])
        new_exits2.append(ind2_newexits[1])
        new_cells2.append(ind1_newcells[1])
        new_cells2.append(ind2_newcells[1])

        if n_guides >= 3:
            new_exits3.append(ind1_newexits[2])
            new_exits3.append(ind2_newexits[2])
            new_cells3.append(ind1_newcells[2])
            new_cells3.append(ind2_newcells[2])

            if n_guides >= 4:
                new_exits4.append(ind1_newexits[3])
                new_exits4.append(ind2_newexits[3])
                new_cells4.append(ind1_newcells[3])
                new_cells4.append(ind2_newcells[3])

                if n_guides >= 5:
                    new_exits5.append(ind1_newexits[4])
                    new_exits5.append(ind2_newexits[4])
                    new_cells5.append(ind1_newcells[4])
                    new_cells5.append(ind2_newcells[4])

                    if n_guides >= 6:
                        new_exits6.append(ind1_newexits[5])
                        new_exits6.append(ind2_newexits[5])
                        new_cells6.append(ind1_newcells[5])
                        new_cells6.append(ind2_newcells[5])

                        if n_guides >= 7:
                            new_exits7.append(ind1_newexits[6])
                            new_exits7.append(ind2_newexits[6])
                            new_cells7.append(ind1_newcells[6])
                            new_cells7.append(ind2_newcells[6])

                            if n_guides >= 8:
                                new_exits8.append(ind1_newexits[7])
                                new_exits8.append(ind2_newexits[7])
                                new_cells8.append(ind1_newcells[7])
                                new_cells8.append(ind2_newcells[7])

                                if n_guides >= 9:
                                    new_exits9.append(ind1_newexits[8])
                                    new_exits9.append(ind2_newexits[8])
                                    new_cells9.append(ind1_newcells[8])
                                    new_cells9.append(ind2_newcells[8])

                                    if n_guides >= 10:
                                        new_exits10.append(ind1_newexits[9])
                                        new_exits10.append(ind2_newexits[9])
                                        new_cells10.append(ind1_newcells[9])
                                        new_cells10.append(ind2_newcells[9])

# Make copies of exits and cells for all next generation simulations
all_new_exits1 = []
all_new_cells1 = []

if n_guides >= 2:
    all_new_exits2 = []
    all_new_cells2 = []

    if n_guides >= 3:
        all_new_exits3 = []
        all_new_cells3 = []

        if n_guides >= 4:
            all_new_exits4 = []
            all_new_cells4 = []

            if n_guides >= 5:
                all_new_exits5 = []
                all_new_cells5 = []

                if n_guides >= 6:
                    all_new_exits6 = []
                    all_new_cells6 = []

                    if n_guides >= 7:
                        all_new_exits7 = []
                        all_new_cells7 = []

                        if n_guides >= 8:
                            all_new_exits8 = []
                            all_new_cells8 = []

                            if n_guides >= 9:
                                all_new_exits9 = []
                                all_new_cells9 = []

                                if n_guides >= 10:
                                    all_new_exits10 = []
                                    all_new_cells10 = []

for i in range(population):
    for j in range(scenarios):
        all_new_exits1.append(new_exits1[i])
        all_new_cells1.append(new_cells1[i])

        if n_guides >= 2:
            all_new_exits2.append(new_exits2[i])
            all_new_cells2.append(new_cells2[i])

            if n_guides >= 3:
                all_new_exits3.append(new_exits3[i])
                all_new_cells3.append(new_cells3[i])

                if n_guides >= 4:
                    all_new_exits4.append(new_exits4[i])
                    all_new_cells4.append(new_cells4[i])

                    if n_guides >= 5:
                        all_new_exits5.append(new_exits5[i])
                        all_new_cells5.append(new_cells5[i])

                        if n_guides >= 6:
                            all_new_exits6.append(new_exits6[i])
                            all_new_cells6.append(new_cells6[i])

                            if n_guides >= 7:
                                all_new_exits7.append(new_exits7[i])
                                all_new_cells7.append(new_cells7[i])

                                if n_guides >= 8:
                                    all_new_exits8.append(new_exits8[i])
                                    all_new_cells8.append(new_cells8[i])

                                    if n_guides >= 9:
                                        all_new_exits9.append(new_exits9[i])
                                        all_new_cells9.append(new_cells9[i])

                                        if n_guides >= 10:
                                            all_new_exits10.append(new_exits10[i])
                                            all_new_cells10.append(new_cells10[i])


# Store the data of the individuals selected for breeding
nondominated_evactimes = []
nondominated_positions = []
nondominated_expected_evactimes = []
nondominated_cvar_evactimes = []

nondominated_exits1 = []
nondominated_cells1 = []

if n_guides >= 2:
    nondominated_exits2 = []
    nondominated_cells2 = []

    if n_guides >= 3:
        nondominated_exits3 = []
        nondominated_cells3 = []

        if n_guides >= 4:
            nondominated_exits4 = []
            nondominated_cells4 = []

            if n_guides >= 5:
                nondominated_exits5 = []
                nondominated_cells5 = []

                if n_guides >= 6:
                    nondominated_exits6 = []
                    nondominated_cells6 = []

                    if n_guides >= 7:
                        nondominated_exits7 = []
                        nondominated_cells7 = []

                        if n_guides >= 8:
                            nondominated_exits8 = []
                            nondominated_cells8 = []

                            if n_guides >= 9:
                                nondominated_exits9 = []
                                nondominated_cells9 = []

                                if n_guides >= 10:
                                    nondominated_exits10 = []
                                    nondominated_cells10 = []



for i in range(population):

    nondominated_expected_evactimes.append(objectives1[next_parent[i]])
    nondominated_cvar_evactimes.append(objectives2[next_parent[i]])

    # Remember that the cells and exits contain each values several times (the amount is equal to scenarios)
    nondominated_exits1.append(exits1[next_parent[i]])
    nondominated_cells1.append(cells1[next_parent[i]])

    if n_guides >= 2:
        nondominated_exits2.append(exits2[next_parent[i]])
        nondominated_cells2.append(cells2[next_parent[i]])

        if n_guides >= 3:
            nondominated_exits3.append(exits3[next_parent[i]])
            nondominated_cells3.append(cells3[next_parent[i]])

            if n_guides >= 4:
                nondominated_exits4.append(exits4[next_parent[i]])
                nondominated_cells4.append(cells4[next_parent[i]])

                if n_guides >= 5:
                    nondominated_exits5.append(exits5[next_parent[i]])
                    nondominated_cells5.append(cells5[next_parent[i]])

                    if n_guides >= 6:
                        nondominated_exits6.append(exits6[next_parent[i]])
                        nondominated_cells6.append(cells6[next_parent[i]])

                        if n_guides >= 7:
                            nondominated_exits7.append(exits7[next_parent[i]])
                            nondominated_cells7.append(cells7[next_parent[i]])

                            if n_guides >= 8:
                                nondominated_exits8.append(exits8[next_parent[i]])
                                nondominated_cells8.append(cells8[next_parent[i]])

                                if n_guides >= 9:
                                    nondominated_exits9.append(exits9[next_parent[i]])
                                    nondominated_cells9.append(cells9[next_parent[i]])

                                    if n_guides >= 10:
                                        nondominated_exits10.append(exits10[next_parent[i]])
                                        nondominated_cells10.append(cells10[next_parent[i]])

    for j in range(scenarios):
        nondominated_evactimes.append(all_evactimes[next_parent[i]*scenarios+j])
        nondominated_positions.append(all_positions[next_parent[i]*scenarios+j])

# Save data of the individuals that survived nondominated sorting

# Save total evacuation times
np.savetxt("{}{}{}".format('nondominated_evactimes_', generation, '.txt'), np.asarray(nondominated_evactimes), fmt='%.14f')

# Save the expected evacuation times
np.savetxt("{}{}{}".format('nondominated_expected_evactimes_', generation, '.txt'), np.asarray(nondominated_expected_evactimes), fmt='%.14f')

# Save the variance of evacuation times
np.savetxt("{}{}{}".format('nondominated_cvar_evactimes_', generation, '.txt'), np.asarray(nondominated_cvar_evactimes), fmt='%.14f')

# Save the positions of all the runs
f = open("{}{}{}".format('nondominated_positions_', generation, '.txt'), "w")
for i in range(n_simulations):
    f.write("{}{}".format(nondominated_positions[i], '\n'))
f.close()

# Save the exits and cells of the selected individuals
f = open("{}{}{}".format('nondominated_', generation, '.txt'), "w")
for i in range(population):
    if n_guides == 1:
        f.write("{}{}{}{}".format(nondominated_exits1[i], ' ', nondominated_cells1[i], '\n'))
    if n_guides == 2:
        f.write("{}{}{}{}{}{}{}{}".format(
                                        nondominated_exits1[i], ' ', nondominated_cells1[i], ' ',
                                        nondominated_exits2[i], ' ', nondominated_cells2[i], '\n'))
    if n_guides == 3:
        f.write("{}{}{}{}{}{}{}{}{}{}{}{}".format(
                                        nondominated_exits1[i], ' ', nondominated_cells1[i], ' ',
                                        nondominated_exits2[i], ' ', nondominated_cells2[i], ' ',
                                        nondominated_exits3[i], ' ', nondominated_cells3[i], '\n'))
    if n_guides == 4:
        f.write("{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}".format(
                                        nondominated_exits1[i], ' ', nondominated_cells1[i], ' ',
                                        nondominated_exits2[i], ' ', nondominated_cells2[i], ' ',
                                        nondominated_exits3[i], ' ', nondominated_cells3[i], ' ',
                                        nondominated_exits4[i], ' ', nondominated_cells4[i], '\n'))
    if n_guides == 5:
        f.write("{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}".format(
                                        nondominated_exits1[i], ' ', nondominated_cells1[i], ' ',
                                        nondominated_exits2[i], ' ', nondominated_cells2[i], ' ',
                                        nondominated_exits3[i], ' ', nondominated_cells3[i], ' ',
                                        nondominated_exits4[i], ' ', nondominated_cells4[i], ' ',
                                        nondominated_exits5[i], ' ', nondominated_cells5[i], '\n'))
    if n_guides == 6:
        f.write("{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}".format(
                                        nondominated_exits1[i], ' ', nondominated_cells1[i], ' ',
                                        nondominated_exits2[i], ' ', nondominated_cells2[i], ' ',
                                        nondominated_exits3[i], ' ', nondominated_cells3[i], ' ',
                                        nondominated_exits4[i], ' ', nondominated_cells4[i], ' ',
                                        nondominated_exits5[i], ' ', nondominated_cells5[i], ' ',
                                        nondominated_exits6[i], ' ', nondominated_cells6[i], '\n'))
    if n_guides == 7:
        f.write(
            "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}".format(
                                    nondominated_exits1[i], ' ', nondominated_cells1[i], ' ',
                                    nondominated_exits2[i], ' ', nondominated_cells2[i], ' ',
                                    nondominated_exits3[i], ' ', nondominated_cells3[i], ' ',
                                    nondominated_exits4[i], ' ', nondominated_cells4[i], ' ',
                                    nondominated_exits5[i], ' ', nondominated_cells5[i], ' ',
                                    nondominated_exits6[i], ' ', nondominated_cells6[i], ' ',
                                    nondominated_exits7[i], ' ', nondominated_cells7[i], '\n'))
    if n_guides == 8:
        f.write(
            "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}".format(
                                    nondominated_exits1[i], ' ', nondominated_cells1[i], ' ',
                                    nondominated_exits2[i], ' ', nondominated_cells2[i], ' ',
                                    nondominated_exits3[i], ' ', nondominated_cells3[i], ' ',
                                    nondominated_exits4[i], ' ', nondominated_cells4[i], ' ',
                                    nondominated_exits5[i], ' ', nondominated_cells5[i], ' ',
                                    nondominated_exits6[i], ' ', nondominated_cells6[i], ' ',
                                    nondominated_exits7[i], ' ', nondominated_cells7[i], ' ',
                                    nondominated_exits8[i], ' ', nondominated_cells8[i], '\n'))
    if n_guides == 9:
        f.write(
            "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}".format(
                                    nondominated_exits1[i], ' ', nondominated_cells1[i], ' ',
                                    nondominated_exits2[i], ' ', nondominated_cells2[i], ' ',
                                    nondominated_exits3[i], ' ', nondominated_cells3[i], ' ',
                                    nondominated_exits4[i], ' ', nondominated_cells4[i], ' ',
                                    nondominated_exits5[i], ' ', nondominated_cells5[i], ' ',
                                    nondominated_exits6[i], ' ', nondominated_cells6[i], ' ',
                                    nondominated_exits7[i], ' ', nondominated_cells7[i], ' ',
                                    nondominated_exits8[i], ' ', nondominated_cells8[i], ' ',
                                    nondominated_exits9[i], ' ', nondominated_cells9[i], '\n'))
    if n_guides == 10:
        f.write(
            "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}".format(
                                    nondominated_exits1[i], ' ', nondominated_cells1[i], ' ',
                                    nondominated_exits2[i], ' ', nondominated_cells2[i], ' ',
                                    nondominated_exits3[i], ' ', nondominated_cells3[i], ' ',
                                    nondominated_exits4[i], ' ', nondominated_cells4[i], ' ',
                                    nondominated_exits5[i], ' ', nondominated_cells5[i], ' ',
                                    nondominated_exits6[i], ' ', nondominated_cells6[i], ' ',
                                    nondominated_exits7[i], ' ', nondominated_cells7[i], ' ',
                                    nondominated_exits8[i], ' ', nondominated_cells8[i], ' ',
                                    nondominated_exits9[i], ' ', nondominated_cells9[i], '0',
                                    nondominated_exits10[i], ' ', nondominated_cells10[i], '\n'))
f.close()


np.savetxt("{}{}{}".format('nondominated_exits1_', generation, '.txt'), np.asarray(nondominated_exits1, dtype=int), fmt='%i', delimiter=' ', newline=' ')
np.savetxt("{}{}{}".format('nondominated_cells1_', generation, '.txt'), np.asarray(nondominated_cells1, dtype=int), fmt='%i', delimiter=' ', newline=' ')
if n_guides >= 2:
    np.savetxt("{}{}{}".format('nondominated_exits2_', generation, '.txt'), np.asarray(nondominated_exits2, dtype=int), fmt='%i', delimiter=' ', newline=' ')
    np.savetxt("{}{}{}".format('nondominated_cells2_', generation, '.txt'), np.asarray(nondominated_cells2, dtype=int), fmt='%i', delimiter=' ', newline=' ')
    
    if n_guides >= 3:    
        np.savetxt("{}{}{}".format('nondominated_exits3_', generation, '.txt'), np.asarray(nondominated_exits3, dtype=int), fmt='%i', delimiter=' ', newline=' ')
        np.savetxt("{}{}{}".format('nondominated_cells3_', generation, '.txt'), np.asarray(nondominated_cells3, dtype=int), fmt='%i', delimiter=' ', newline=' ')
        
        if n_guides >= 4:
            np.savetxt("{}{}{}".format('nondominated_exits4_', generation, '.txt'), np.asarray(nondominated_exits4, dtype=int), fmt='%i', delimiter=' ', newline=' ')
            np.savetxt("{}{}{}".format('nondominated_cells4_', generation, '.txt'), np.asarray(nondominated_cells4, dtype=int), fmt='%i', delimiter=' ', newline=' ')
            
            if n_guides >= 5:
                np.savetxt("{}{}{}".format('nondominated_exits5_', generation, '.txt'), np.asarray(nondominated_exits5, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                np.savetxt("{}{}{}".format('nondominated_cells5_', generation, '.txt'), np.asarray(nondominated_cells5, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                
                if n_guides >= 6:
                    np.savetxt("{}{}{}".format('nondominated_exits6_', generation, '.txt'), np.asarray(nondominated_exits6, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                    np.savetxt("{}{}{}".format('nondominated_cells6_', generation, '.txt'), np.asarray(nondominated_cells6, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                    
                    if n_guides >= 7:
                        np.savetxt("{}{}{}".format('nondominated_exits7_', generation, '.txt'), np.asarray(nondominated_exits7, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                        np.savetxt("{}{}{}".format('nondominated_cells7_', generation, '.txt'), np.asarray(nondominated_cells7, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                        
                        if n_guides >= 8:
                            np.savetxt("{}{}{}".format('nondominated_exits8_', generation, '.txt'), np.asarray(nondominated_exits8, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                            np.savetxt("{}{}{}".format('nondominated_cells8_', generation, '.txt'), np.asarray(nondominated_cells8, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                            
                            if n_guides >= 9:
                                np.savetxt("{}{}{}".format('nondominated_exits9_', generation, '.txt'), np.asarray(nondominated_exits9, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                                np.savetxt("{}{}{}".format('nondominated_cells9_', generation, '.txt'), np.asarray(nondominated_cells9, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                                
                                if n_guides >= 10:
                                    np.savetxt("{}{}{}".format('nondominated_exits10_', generation, '.txt'), np.asarray(nondominated_exits10, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                                    np.savetxt("{}{}{}".format('nondominated_cells10_', generation, '.txt'), np.asarray(nondominated_cells10, dtype=int), fmt='%i', delimiter=' ', newline=' ')


# Save data of the offspring
np.savetxt("{}{}{}".format('exits1_', generation+1, '.txt'), np.asarray(all_new_exits1, dtype=int), fmt='%i', delimiter=' ', newline=' ')
np.savetxt("{}{}{}".format('cells1_', generation+1, '.txt'), np.asarray(all_new_cells1, dtype=int), fmt='%i', delimiter=' ', newline=' ')
if n_guides >= 2:
    np.savetxt("{}{}{}".format('exits2_', generation+1, '.txt'), np.asarray(all_new_exits2, dtype=int), fmt='%i', delimiter=' ', newline=' ')
    np.savetxt("{}{}{}".format('cells2_', generation+1, '.txt'), np.asarray(all_new_cells2, dtype=int), fmt='%i', delimiter=' ', newline=' ')
    
    if n_guides >= 3:
        np.savetxt("{}{}{}".format('exits3_', generation+1, '.txt'), np.asarray(all_new_exits3, dtype=int), fmt='%i', delimiter=' ', newline=' ')
        np.savetxt("{}{}{}".format('cells3_', generation+1, '.txt'), np.asarray(all_new_cells3, dtype=int), fmt='%i', delimiter=' ', newline=' ')
        
        if n_guides >=4:
            np.savetxt("{}{}{}".format('exits4_', generation+1, '.txt'), np.asarray(all_new_exits4, dtype=int), fmt='%i', delimiter=' ', newline=' ')
            np.savetxt("{}{}{}".format('cells4_', generation+1, '.txt'), np.asarray(all_new_cells4, dtype=int), fmt='%i', delimiter=' ', newline=' ')
            
            if n_guides >= 5:
                np.savetxt("{}{}{}".format('exits5_', generation+1, '.txt'), np.asarray(all_new_exits5, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                np.savetxt("{}{}{}".format('cells5_', generation+1, '.txt'), np.asarray(all_new_cells5, dtype=int), fmt='%i', delimiter=' ', newline=' ')
            
                if n_guides >= 6:
                    np.savetxt("{}{}{}".format('exits6_', generation+1, '.txt'), np.asarray(all_new_exits6, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                    np.savetxt("{}{}{}".format('cells6_', generation+1, '.txt'), np.asarray(all_new_cells6, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                    
                    if n_guides >= 7:
                        np.savetxt("{}{}{}".format('exits7_', generation+1, '.txt'), np.asarray(all_new_exits7, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                        np.savetxt("{}{}{}".format('cells7_', generation+1, '.txt'), np.asarray(all_new_cells7, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                        
                        if n_guides >= 8:
                            np.savetxt("{}{}{}".format('exits8_', generation+1, '.txt'), np.asarray(all_new_exits8, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                            np.savetxt("{}{}{}".format('cells8_', generation+1, '.txt'), np.asarray(all_new_cells8, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                            
                            if n_guides >= 9:
                                np.savetxt("{}{}{}".format('exits9_', generation+1, '.txt'), np.asarray(all_new_exits9, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                                np.savetxt("{}{}{}".format('cells9_', generation+1, '.txt'), np.asarray(all_new_cells9, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                                
                                if n_guides >= 10:
                                    np.savetxt("{}{}{}".format('exits10_', generation+1, '.txt'), np.asarray(all_new_exits10, dtype=int), fmt='%i', delimiter=' ', newline=' ')
                                    np.savetxt("{}{}{}".format('cells10_', generation+1, '.txt'), np.asarray(all_new_cells10, dtype=int), fmt='%i', delimiter=' ', newline=' ')
