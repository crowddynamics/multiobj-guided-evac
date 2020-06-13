import numpy as np
import matplotlib.pyplot as plt
import random

from nsga2 import nondominatedsort, crowdingdistance
from hypervolume import hypervolume

# Change the last_generation and nadir_point appropriate to the problem
first_generation = 0
last_generation = 33
generations = [first_generation+i for i in range(last_generation-first_generation+1)]
nadir_point = [271, 271]
for generation in generations:

    objectives1 = np.loadtxt("{}{}{}".format('nondominated_expected_evactimes_', generation, '.txt'))
    #objectives2 = np.loadtxt("{}{}{}".format('nondominated_variance_evactimes_', generation, '.txt'))
    objectives2 = np.loadtxt("{}{}{}".format('nondominated_cvar_evactimes_', generation, '.txt'))

    fronts, rank = nondominatedsort(objectives1, objectives2, "min", "min")
    print(fronts)

    color = ['black', 'yellow', 'blue', 'black', 'yellow', 'blue', 'black', 'yellow', 'blue', 'black', 'yellow', 'blue', 'black', 'yellow', 'blue', 'black', 'yellow', 'blue', 'black', 'yellow', 'blue']
    marker = ["o", "o", "o", "v", "v", "v", "s", "s", "s", "+", "+", "+", ".", ".", ".", "*", "*", "*", "<", "<", "<"]

    f2 = plt.figure(generation)
    for i in range(len(fronts)):
        #if i>0:
        #    continue
        plot_data_x = []
        plot_data_y = []
        for j in range(len(fronts[i])):
            plot_data_x.append(objectives1[fronts[i][j]])
            plot_data_y.append(objectives2[fronts[i][j]])          
            #plot_data_y.append(np.sqrt(objectives2[fronts[i][j]]))
        plt.scatter(plot_data_x, plot_data_y, c = color[i], marker=marker[i])
        if i == 0:
            objectives = np.vstack((plot_data_x, plot_data_y))
            unique_columns, indices = np.unique(objectives, axis=1, return_inverse=True)
            unique_objectives1 = unique_columns[0]
            unique_objectives2 = unique_columns[1]
            hyperv, cardinality = hypervolume(unique_objectives1, unique_objectives2, nadir_point)
            print(hyperv)
        if i == 0:
            for j in range(len(fronts[i])):
                txt = "{}{}{}".format(round(objectives1[fronts[i][j]], 2), ', ', round(objectives2[fronts[i][j]], 2))
                plt.annotate(txt, (objectives1[fronts[i][j]], objectives2[fronts[i][j]]+1), size=4)
    plt.ylim(-5,1000)
    plt.xlim(-5,200)
    plt.xlabel('Expected evacuation time (s)')
    plt.ylabel('Standard deviation (s)')
    plt.title("{}{}{}{}{}{}".format('generation ', generation, ', hypervolume ', round(hyperv, 2), ', cardinality ', cardinality))

    plt.savefig("{}{}{}".format('fronts_nondominated_', generation, '.png'))
