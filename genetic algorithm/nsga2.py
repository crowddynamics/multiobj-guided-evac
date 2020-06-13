import numpy as np

# sort the solutions in pareto fronts
def nondominatedsort(objectives1, objectives2, str1, str2):
    P = len(objectives1) # number of solutions
    S = [] # sets of solutions dominated by a solution
    n = np.zeros(P) # domination counter for the solutions
    rank = np.ones(P) # which front a solution belongs to
    F = [] # fronts
    F.append([]) # first front is initially empty
    for p in range(P):
        S.append([]) # set of solutions dominated by p is initially empty
        #n.append(0) # domination counter for p is initially zerro
        for q in range(P):
            # NOTE: THESE DEPEND ON WHETHER WE ARE MINIMIZING OR MAXIMIZING THE OBJECTIVES
            # if p dominates q, add q to the set of solutions dominated by p
            # else if q dominates p, increment the domination counter of p
            if str1 == "max" and str2 == "max":

                if ((objectives1[p] >= objectives1[q]) and (objectives2[p] > objectives2[q])) or ((objectives1[p] > objectives1[q]) and (objectives2[p] >= objectives2[q])):
                    S[p].append(q)
                elif ((objectives1[q] >= objectives1[p]) and (objectives2[q] > objectives2[p])) or ((objectives1[q] > objectives1[p]) and (objectives2[q] >= objectives2[p])):
                    n[p] += 1

            elif str1 == "max" and str2 == "min":

                if ((objectives1[p] >= objectives1[q]) and (objectives2[p] < objectives2[q])) or ((objectives1[p] > objectives1[q]) and (objectives2[p] <= objectives2[q])):
                    S[p].append(q)
                elif ((objectives1[q] >= objectives1[p]) and (objectives2[q] < objectives2[p])) or ((objectives1[q] > objectives1[p]) and (objectives2[q] <= objectives2[p])):
                    n[p] += 1

            elif str1 == "min" and str2 == "max":

                if ((objectives1[p] <= objectives1[q]) and (objectives2[p] > objectives2[q])) or ((objectives1[p] < objectives1[q]) and (objectives2[p] >= objectives2[q])):
                    S[p].append(q)
                elif ((objectives1[q] <= objectives1[p]) and (objectives2[q] > objectives2[p])) or ((objectives1[q] < objectives1[p]) and (objectives2[q] >= objectives2[p])):
                    n[p] += 1

            elif str1 == "min" and str2 == "min":

                if ((objectives1[p] <= objectives1[q]) and (objectives2[p] < objectives2[q])) or ((objectives1[p] < objectives1[q]) and (objectives2[p] <= objectives2[q])):
                    S[p].append(q)
                elif ((objectives1[q] <= objectives1[p]) and (objectives2[q] < objectives2[p])) or ((objectives1[q] < objectives1[p]) and (objectives2[q] <= objectives2[p])):
                    n[p] += 1

        # p belongs to the first front
        if n[p] == 0:
            F[0].append(p)

    # initialize the front counter
    i = 1
    #print(F[i-1])
    while len(F[i-1]) != 0:
        # used to store the members of the next front
        Q = []
        for p in F[i-1]:
            #print(p)
            for q in S[p]:
                #print(q)
                n[q] -= 1
                # q belongs to the next front
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        if len(Q) != 0:
            F.append([])
            F[i-1] = Q
        else:
            break

    return F, rank

# calculate crowding distances for a front
def crowdingdistance(objectives1, objectives2, str1, str2):
    
    # 1) Deb, K., Pratap, A., Agarwal, S., and Meyarivan T. A fast and elitist multiopbjective genetic algorithm: NSGA-II.
    # 2) Fortin, F.-A., Parizeau, M. Revisiting the NSGA-II crowding-distance computation.
    #
    # A) The crowding-distance computation remains unchanged. However, the provided inputs is replaced with unique fitnesses
    # instead of individuals.
    # B) Once, the crowding distance is computed for the unique fitnesses, the distances are assigned to individuals with the
    # corresponding fitness.
    # C) With this approach, every individual sharing the same fitness inherits the same crowding distance and therefore the
    # minimum crowding distance is always greater than 0.
    # D) The crowding distance calculated this way represents a correct density estimation in the objective space when
    # individuals share the same fitness.
    # F) The advantage of the previous formulation was that some individuals with equal fitnesses could be assigned a crowding
    # distance of, limiting their chances of reproduction and therefore stimulating diversity preservation. However, we
    # argue that the diversity preservation goal should not be accomplished with a biased metric, but a correct
    # interpretation of the metric through the selection process.

    # Number of individuals
    n_individuals = len(objectives1)

    # create a 2-d array of objectivevalues
    objectives = np.vstack((objectives1, objectives2))
    # return unique columns, and an array with same indices for elements with same columns
    unique_columns, indices = np.unique(objectives, axis=1, return_inverse=True)
    # take the minimal number of indices that give the unique columns
    meta_unique_indices = np.unique(indices)
    meta_unique_indices.tolist()
    unique_indices = [indices[i] for i in meta_unique_indices]
    
    unique_objectives1 = unique_columns[0]
    unique_objectives2 = unique_columns[1]

    # number of unique solutions
    n_unique = len(unique_objectives1)

    if n_unique > 2:

        # initialize unique distances to zero
        unique_distance = np.zeros(n_unique)

        # maximum of objective 1
        max_1 = np.max(unique_objectives1)
        # minimum of objective 1
        min_1 = np.min(unique_objectives1)
        # maximum of objective 2
        max_2 = np.max(unique_objectives2)
        # minimum of objective 2
        min_2 = np.min(unique_objectives2)

        # Technically not possible?
        # Because if max_1 == min_1 and max_2 == min_2 => all points are the same (and n_unique==1)
        # if just max_1 == min_1 OR max_2 == min_2 => the points should not be on the same front!
        if max_1 == min_1 or max_2 == min_2:
            distance = np.ones(n_individuals) * 10000
            return distance

        # We need just to sort according to the first objective value.
        # The below operation gives us the indices of solutions from smallest to largest w.r.t. objective 1
        obj1_args = sorted(range(n_unique), key=lambda k : unique_objectives1[k])
        # The endpoints get large distance values so that they are always chosen
        unique_distance[obj1_args[0]] = 10000 # rather than giving the value infinity, we just give a large number
        unique_distance[obj1_args[-1]] = 10000

        # Loop through the rest of the points and calculate their distances.
        # NOTE: THESE DEPEND ON WHETHER WE ARE MINIMIZING OR MAXIMIZING THE OBJECTIVES
        if str1 == "max" and str2 == "max":
            for i in range(1,n_unique-1):
                unique_distance[obj1_args[i]] += (unique_objectives1[obj1_args[i+1]] - unique_objectives1[obj1_args[i-1]]) / (max_1 - min_1) +\
                                          (unique_objectives2[obj1_args[i-1]] - unique_objectives2[obj1_args[i+1]]) / (max_2 - min_2)

        elif str1 == "max" and str2 == "min":
            for i in range(1,n_unique-1):
                unique_distance[obj1_args[i]] += (unique_objectives1[obj1_args[i+1]] - unique_objectives1[obj1_args[i-1]]) / (max_1 - min_1) +\
                                          (unique_objectives2[obj1_args[i+1]] - unique_objectives2[obj1_args[i-1]]) / (max_2 - min_2)

        elif str1 == "min" and str2 == "max":
            for i in range(1,n_unique-1):
                unique_distance[obj1_args[i]] += (unique_objectives1[obj1_args[i+1]] - unique_objectives1[obj1_args[i-1]]) / (max_1 - min_1) +\
                                          (unique_objectives2[obj1_args[i+1]] - unique_objectives2[obj1_args[i-1]]) / (max_2 - min_2)

        elif str1 == "min" and str2 == "min":
            for i in range(1,n_unique-1):
                unique_distance[obj1_args[i]] += (unique_objectives1[obj1_args[i+1]] - unique_objectives1[obj1_args[i-1]]) / (max_1 - min_1) +\
                                          (unique_objectives2[obj1_args[i-1]] - unique_objectives2[obj1_args[i+1]]) / (max_2 - min_2)

        # B) Once, the crowding distance is computed for the unique fitnesses, the distances are assigned to individuals with the
        # corresponding fitness.
        # C) With this approach, every individual sharing the same fitness inherits the same crowding distance and therefore the
        # minimum crowding distance is always greater than 0.

        # initialize distances to zero
        distance = np.zeros(n_individuals)

        for i in range(n_unique):
            similars = np.where(indices==unique_indices[i])
            for j in range(len(similars)):
                distance[similars[j]] = unique_distance[i]

    else:
        distance = np.ones(n_individuals) * 10000
        
    # The crowding distances should be returned in the same order of the input solutions
    return distance

# compare two individuals
def crowdedcomparison(competitor1, competitor2, rank1,  rank2, distance1, distance2):
     # if rank1 < rank2, 1 dominates 2
     if rank1 < rank2:
         return competitor2
     # if rank2 < rank1, 2 dominates 1
     elif rank2 < rank1:
         return competitor1
     # if ranks are equal, the solutions are on the same front
     else:
         # if distance1 > distance2, 1 dominates 2
         if distance1 > distance2:
             return competitor1
         # if distance2 > distance1, 2 dominates 1
         elif distance2 > distance1:
             return competitor2
         # the solutions are the same, return 1
         else:
             return competitor1
