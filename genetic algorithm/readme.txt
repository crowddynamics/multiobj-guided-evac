Here are instructions for using the genetic algorithm codes. NOTE! The specifications for the bash script files have to probably be changed depending on the computational cluster being used.

grand_scheme.sh
The main script, that calls other scripts to execute the combined numerical simulation and NSGA-II. Here, you specify the number of generations, scenarios and guides, and population size. Due to user quotas on the computational cluster, you have the option to run several simulations on the same node (alter parameter "BATCH_SIZE"), and instead of running all simulations of the population simultaneously, you can run them in parts (alter parameters "PARTS" and "SIMULATIONS_PART"). The procedure is run using a dependency chain, so the algorithm can be stopped and continued.

generate_scenarios.sh
The script creates scenario numbers to be used as input parameters for the crowd simulation.

initialize.sh
Creates randomly the values for the optimization variables (=the guides' starting positions and destination exits) for the 0th generation.

genetic_algorithm_.sh
A bash script that sends the scenario number and values for the optimization variables to the crowd simulation model "shell_run_complex.py".

shell_run_complex.py
Python script for simulating the crowd evacuation.

complex_ga.py
Contains the floor of the building and some other model specifications.

spawn_complex.npy
Data file that includes the initial positions of the passenger agents. We need this so that the guides are not positioned so that they overlap with the passenger agents.

gather_results.sh
Sends the current generation number, population size, number of scenarios and guides, batch size, number of partitions and number of simulations in a partition to "selection.py".

selection.py
A python script that performs nondominated sorting, unique fitness tournament selection, crossover and mutation for one generation.

nsga2.py
Includes three python functions: nondominatedsort, crowdingdistance and crowdedcomparison. These are all used when choosing the parent solutions.

lastfrontselection.py
A python function for performing the last front selection. If the all the solutions in a nondominated front does not fit into the parent population, those solutions that have the highest crowding distance are selected.

uftournselection.py
A python function for performing unique fitness tournament selection.

onepointcrossover.py
A python function for the onepointcrossover operation for the genes.

mutation.py
A python function for the mutation operation for the genes.

plot_script.sh
A script that calls the file "plot_fronts.py".

plot_fronts.py
A python script that plots the nondominated fronts, and calculates the hypervolume indicators for the first nondominated front.

hypervolume.py
A python function for calculating the hypervolume indicator.

solutionbank_complex.py
A python function for storing the genes, scenarios and their associated fitnesses, so that they only have to be simulated once.
