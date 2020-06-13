def hypervolume(objectives1, objectives2, nadir):

    # Cardinality of Pareto front
    cardinality = len(objectives1)

    # Initialize the hypervolume to zero
    value = 0

    # In the bi-objective case, the hypervolume is an area.
    # Calculate the hypervolume, by calculating the sub-areas
    # by looping over the Pareto points.
    for i in range(cardinality):
    
        if i == 0:
            value += (nadir[0]-objectives1[i])*(nadir[1]-objectives2[i])
        else:
            value += (nadir[0]-objectives1[i])*(objectives2[i-1]-objectives2[i])

    return value, cardinality
