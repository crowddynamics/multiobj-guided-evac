import os

def solutionbank(cells, exits, n_leaders, scenario):
    # Compare the solution with the solutions already evaluated
    
    # Load the solution bank (read line by line)
    fname = "complex/bank_complex.out"
    if ~os.path.isfile(fname):
        return 0

    with open(fname) as infile:
        lines = [line.rstrip('\n') for line in infile]
    
    # Loop through the lines in the solution bank file to gather solutions
    # The first number is the total evacuation time, the second
    # the scenario number, and the rest are the cells and exits for each gene       
    n_simulations = len(lines)
    for i in range(n_simulations):
        solution = lines[i].split(" ")
        solution = [float(solution[j]) for j in range(len(solution))]
        
        # Check if the scenario number of the solution in the bank equals the scenario of the input solution
        if scenario == solution[1]:

            # Check if the solution from the bank equals the input solution
            solution_length = (len(solution)-2)/2
            if solution_length == n_leaders:

                 # Check if the solution in the bank matches the input solution
                 n_matches = 0

                 # Check if the solution in the bank has the gene (cells[i], exits[i])
                 for j in range(solution_length):
                     for k in range(n_leaders):
                         if cells[k] == solution[2+2*j] and exits[k] == solution[3+2*j]:
                             n_matches += 1

                 # If the solution in the bank matches the input solution, return the total evacuation time
                 if n_matches == n_leaders:
                     return solution[0]

    return 0
