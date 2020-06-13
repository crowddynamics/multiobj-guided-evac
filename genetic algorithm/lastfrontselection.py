import numpy as np

# The procedure takes two arguments: a front \mathcal{F} and the number of solutions to select k.
def lastfrontselection(objectives1, objectives2, distances, k):
    
    # The algorithm starts by sorting the unique fitnesses associated with the members of the front
    # \mathcal{F} in descending order of crowding distance (>_{dist}) and stores the result in a list F

    # Number of individuals in the front
    N = len(objectives1)

    # Give same ordinal numbers for the individuals with same objective values (fitnesses)
    objectives = np.vstack((objectives1, objectives2))
    unique_fitnesses, ordinals = np.unique(objectives, axis=1, return_inverse=True)

    # Sort the crowding distances in descending order, and return the indices of the individuals
    tmp_indxs = sorted(range(len(distances)),key=distances.__getitem__)
    tmp_indxs = tmp_indxs[::-1]

    # Sort the individuals' in descending order of crowding distance
    distance_sorted_fitnesses = []
    for i in range(N):
        distance_sorted_fitnesses.append(ordinals[tmp_indxs[i]])

    # Return the unique ordinal numbers (same fitnesses should have same crowding distance)
    unique_ordinals = []
    for i in range(N):
        if i == 0 or  distance_sorted_fitnesses[i-1] != distance_sorted_fitnesses[i]:
            unique_ordinals.append(distance_sorted_fitnesses[i])

    # It then proceeds to fill the initially empty set of solutions S by cycling over the sorted unique
    # fitnesses
    S = []
    j = 0
    while len(S) != k:
        # For each fitness, the algorithm first selects each individual sharing this fitness and puts
        # the result in set T
        T = np.where(np.asarray(ordinals==unique_ordinals[j]))
        T = T[0]

        # Remove from T elements that are already in S
        if len(T) != 0:
            if len(S) != 0:
                for m in range(len(S)):
                    old_chosen = np.where(T==S[m])
                    old_chosen = old_chosen[0]
                    if len(old_chosen) != 0:
                        for n in range(len(old_chosen)):
                            del_element = old_chosen[n]
                            T = np.delete(T, del_element)        

        # If the resulting set is not empty, the procedure randomly picks one solution from T, adds it
        # to set S and finally removes it from the front \mathcal{}F so that individuals can be picked
        #  only once.
        if len(T) != 0:
            choice = np.random.choice(T, 1)
            choice = choice[0]
            S.append(choice)             

        if j == len(unique_ordinals) - 1:
            j = 0
        else:
            j = j + 1

    # When all k solutions have been selected, the loop stops and set S is returned.
    return S

