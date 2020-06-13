import numpy as np


def mutation(exits, locations, mutpb):
    """Mutation that alters the routes of the guides of a single chromosome."""

    # Feasible starting cells
    # The space the agents are evacuating from has been divided into 3m x 3m cells, and we have checked which of these are feasible.
    cells =  np.load('feasible_cells_complex.npy')

    # Feasible exits
    feasible_exits = np.array([0, 1, 2, 3])

    # Number of guides/genes
    n_guides = len(locations)
    # Number of exits
    n_exits = 4

    # Loop through genes of the chromosome and mutate each with probability mutpb.
    for i in range(n_guides):

        if np.random.rand(1)[0] <= mutpb:

            # Either alter the initial location or exit of the guide (with equal probability)
            cell_or_exit = np.random.rand(1)[0]

            if cell_or_exit > 0.5:

                # Move the guide's location randomly
                delete_element = np.where(cells == locations[i])
                available_cells = np.delete(cells, delete_element, None)
                random_cell = available_cells[np.random.randint(len(available_cells))]

                # Generate a random location from the available locations
                locations[i] = random_cell

            else:

                # Move the guide's exit randomly
                delete_element = np.where(feasible_exits == exits[i])
                available_exits = np.delete(feasible_exits, delete_element, None)
                random_exit = available_exits[np.random.randint(len(available_exits))]

                # Generate a random exit from the available exits
                exits[i] = random_exit 

    return exits, locations
