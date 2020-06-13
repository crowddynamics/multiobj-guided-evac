import numpy as np
from nsga2 import crowdedcomparison

# Unique fitness based tournament selection
def uftournselection(objectives1, objectives2, ranks, distances):
    
    # The operator takes a population I of N individuals, where every fitness has previously been assigned a rank
    # and a crowding distance. The crowded-comparison operator <_c can therefore be used to compare them.
    N = len(objectives1)

    # The operator starts by building a set F of unique fitnesses associated with the individuals.
    #
    # create a 2-d array of objectivevalues
    objectives = np.vstack((objectives1, objectives2))
    # return unique columns, and an array with same indices for elements with same columns
    unique_columns, indices = np.unique(objectives, axis=1, return_inverse=True)
    # take the minimal number of indices that give the unique columns
    meta_unique_indices = np.unique(indices)
    meta_unique_indices.tolist()
    unique_indices = [indices[i] for i in meta_unique_indices]    

    # In the extreme event where every solution in the population shares the same fitness, no selection occurs
    # and the population is simply returned as is.
    n_unique = len(unique_indices)
    if n_unique == 1:
        selected = [i for i in range(40)]
        return selected

    # Otherwise, a set of solutions S is initialized empty, and while the number of selected solutions is not equal to the
    # size of the population, the procedure samples k fitnesses that are stored in a list G.
    S = []
    while len(S) != N:
        # Next we take the list of unique indices and sample k randomly chosen elements without replacement.
        # The number k of sampled fitnesses is the minimum value between the unique fitnesses, corresponding to the
        # cardinality of F, and two times the number of solutions left to select N-|S|.
        k = np.min([2*(N-len(S)), n_unique])
        G = np.random.choice(unique_indices, k, replace=False)
        # The fitnesses are paired and compared using the crowded-comparison operator (<_c).
        for i in range(int(np.floor(len(G)/2))):
            ind1 = G[2*i]
            ind2 = G[2*i+1]
            p=crowdedcomparison(ind1, ind2, ranks[ind1], ranks[ind2], distances[ind1], distances[ind2])
            # A solution associated with the winning fitness is then randomly selected amongst the individuals sharing this
            # fitness and added to set S. After the selection of N solutions, set S is returned.
            temp_indices = np.where(indices==p)
            temp_indices = temp_indices[0]
            choice = np.random.choice(temp_indices, 1)
            choice = choice[0]
            S.append(choice)
    return S
