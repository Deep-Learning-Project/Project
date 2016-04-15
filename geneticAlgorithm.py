import bottleneck as bn
import numpy as np

from Code.graph_reader import parse_graph_file, adjacency_matrix

pop_size = 200
num_gen = 4000
xover_rate = 0.7
mut_rate = 0.01
num_elite = 15


def solve(adj_mat: np.ndarray):
    """
    The basic genetic algorithm. Graph is represented by an adjacency matrix
    :param adj_mat: Adjacency matrix of graph instance required to solve
    :return: The best found vertex cover for the graph.
    """
    # Generate pop_size number of random solutions. Solutions may be infeasible
    population = generate_population(pop_size, adj_mat.shape[0])
    for generation in range(num_gen):
        # Check for infeasible solutions and fix each solution so that it's feasible
        fix_population(population, adj_mat)
        # Calculate the "goodness" of each solution and give it a score
        fitness_scores = evaluate_fitness(population)
        # Introduce concept of elitism. In each generation, num_elite number of best solutions will be chosen to be
        # carried forward to the next iteration without any modifications to them. Instead of sorting entire array,
        # sort it partially so that only top num_elite number of solutions are sorted, and the rest remains same
        if num_elite != 0:
            part_sorted = bn.argpartsort(fitness_scores, fitness_scores.shape[0] - num_elite)
            elite_indices = part_sorted[-num_elite:]  # Sorted top num_elite no of solutions
            remaining_indices = part_sorted[:-num_elite]  # Unsorted solutions
        # Print the best fitness score of every 20 generations to see how algorithm is performing.
        if generation % 100 == 0:
            print(np.max(fitness_scores))
        # Select parents which will create the next generation
        if num_elite != 0:
            parents = select_parents(population[remaining_indices], fitness_scores[remaining_indices])
        else:
            parents = select_parents(population, fitness_scores)
        # Perform a crossover operation on selected parents
        children = crossover(parents, xover_rate, adj_mat)
        # Perform a mutation operation on crossovered parents
        mutate(children, mut_rate)

        if num_elite != 0:
            # Add the elite solutions that were removed from parents, and set them as population (parents) for next gen
            population = np.vstack((children, population[elite_indices]))
        else:
            population = children

    # Fix the final solution obtained to make it feasible.
    fix_population(population, adj_mat)
    # Find solution with highest fitness scores, i.e. solution with least number of vertices.
    best_solution_index = np.argmax(fitness_scores)
    return population[best_solution_index]


def fix_population(population: np.ndarray, adj_mat: np.ndarray):
    """
    Solution generated may not be a complete vertex cover. Check each solution and add vertices to it to make it cover
    all edges
    :param population: 2D array representing a set of solutions for the given graph instance
    :param adj_mat: Adjacency matrix representing the graph instance
    :return: An array of feasible solutions to the graph instance
    """
    # Find vertices which are not connected to vertices currently in cover and add all of them to the cover.
    # unconn_vertices = get_unconnected_vertices(population, adj_mat)
    # adj_matrices = np.matlib.repmat(adj_matrix,
    #                                 population.shape[0], 1).reshape((population.shape[0], *adj_matrix.shape))
    # adj_matrices = (adj_matrices * (1 - population[:, np.newaxis, :])) * (1 - population[:, :, np.newaxis])
    not_population = 1 - population
    unconn_vertices = np.einsum('ij, kij, kij -> kj', adj_mat,
                                not_population[:, np.newaxis, :],
                                not_population[..., np.newaxis])
    # unconn_vertices = adj_matrices.sum(axis=1)
    unconn_vertices[unconn_vertices.nonzero()] = 1
    population += unconn_vertices
    # print("Vertices after fixing are: ", np.sum(population, axis=1))


def generate_population(population_size, num_vertices):
    """
    Generates population_size number of random solutions for the vertex cover. May generate invalid solutions.
    :param population_size: Number of solutions to be generated
    :param num_vertices: Size of graph instance. Required for binary representation of vertex cover
    :return: A 2D ndarray of shape (population_size, num_vertices) where each row indicates a solution to the graph
    instance
    """
    population = np.random.random_integers(0, 1, size=(population_size, num_vertices))
    return population


def evaluate_fitness(population: np.ndarray):
    # Return the number of vertices not in cover
    return population.shape[1] - np.sum(population, axis=1)


def select_parents(population, fitness_scores):
    """
    Use some selection operator to implement the concept of "survival of the fittest" on the solutions
    :param population: A set of solutions to the given graph instance
    :param fitness_scores: Fitness scores of the given solutions
    :return: An array of solutions which are a vague indicator of the best solutions in the population
    """
    # Convert fitness scores to probabilities
    fitness_probabilities = fitness_scores / np.sum(fitness_scores)
    # Do a roulette selection using the probabilities. i.e. parents with higher fitness scores have greater chance of
    # being selected
    parent_indices = np.random.choice(np.arange(fitness_probabilities.shape[0]),
                                      size=population.shape[0],
                                      p=fitness_probabilities)
    return population[parent_indices]


def crossover(parents: np.ndarray, crossover_rate: float = xover_rate, adj_mat=None, xover_type='uox'):
    if xover_type == 'normal':
        xover_rows = np.random.choice([0, 1], size=parents.shape[0] // 2, p=[1 - crossover_rate, crossover_rate])
        xover_rows = np.array(np.repeat(xover_rows, 2).astype(bool))
        xover_parents = parents[xover_rows]
        # Generate random indices for all even rows, and copy them over to odd rows.
        xover_positions = np.random.random_integers(2, parents.shape[1] - 2, size=xover_parents.shape[0] / 2).repeat(2)

        xover_indices = np.repeat(xover_positions.cumsum(),
                                  xover_parents.shape[1] - xover_positions)
        xover_elements = np.insert(np.ones(xover_positions.sum()),
                                   xover_indices,
                                   0).reshape(xover_parents.shape).astype(bool)
        xover_parents_swapped = np.empty(xover_parents.shape)
        xover_parents_swapped[0::2], xover_parents_swapped[1::2] = xover_parents[1::2].copy(), xover_parents[
                                                                                               0::2].copy()
        xover_children = np.empty(xover_parents.shape)
        xover_children[xover_elements], xover_children[~xover_elements] = xover_parents[xover_elements], \
                                                                          xover_parents_swapped[~xover_elements]

        return parents

    elif xover_type == 'uox':
        # Select rows on which crossover is going to be performed
        xover_rows = np.random.choice([False, True], size=parents.shape[0], p=[1 - crossover_rate, crossover_rate])
        if xover_rows.sum() % 2 != 0:
            xover_rows[np.argwhere(xover_rows==False)[0]] = True
        xover_parents = parents[xover_rows]
        xover_bits = np.random.random_integers(0, 1,
                                               size=(xover_parents.shape[0] / 2, xover_parents.shape[1])).repeat(2,
                                                                                                                 axis=0)
        xover_bits[1::2] = 1 - xover_bits[1::2]
        xover_children = np.empty(shape=xover_parents.shape, dtype=xover_parents.dtype)
        xover_parents1, xover_parents2 = xover_parents * xover_bits, xover_parents * (1 - xover_bits)
        xover_children[0::2] = xover_parents1[0::2] + xover_parents1[1::2]
        xover_children[1::2] = xover_parents2[0::2] + xover_parents2[1::2]
        return np.vstack((xover_parents, parents[~xover_rows]))

    elif xover_type == 'other':
        # Select rows on which crossover is going to be performed
        xover_rows = np.random.choice([False, True], size=parents.shape[0], p=[1 - crossover_rate, crossover_rate])
        xover_parents = parents[xover_rows]
        # Find degrees of vertices in cover
        degrees = adj_mat.sum(axis=0)
        parent_degrees = xover_parents * degrees  # Degrees of vertices in cover
        mean_degrees = np.mean(parent_degrees, axis=1)  # Mean degree of each cover
        xover_parents[degrees < mean_degrees[..., np.newaxis]] = 0  # Vertices in cover having lower degree are removed
        # Combine successive parents
        xover_parents[:-1] = xover_parents[:-1] + xover_parents[1:] - (xover_parents[:-1] * xover_parents[1:])
        return np.vstack((xover_parents, parents[~xover_rows]))


def mutate(children: np.ndarray, mutation_rate):
    for row in range(0, children.shape[0], 2):
        if np.random.binomial(1, mutation_rate):
            mut_pos = np.random.random_integers(1, children.shape[1] - 1)
            children[row, mut_pos] = 1 - children[row, mut_pos]


def get_unconnected_vertices(population: np.ndarray, adj_mat: np.ndarray):
    # Get connected vertices:
    conn_vertices = np.transpose(np.triu(adj_mat) @ population.T)
    # Get current vertices and connected vertices (union):
    # print("Vertices before fixing are: ", np.sum(population, axis=1))
    conn_vertices += population

    conn_vertices[conn_vertices.nonzero()] = 1
    return 1 - conn_vertices


def num_edges_not_in_cover(population: np.ndarray, adj_mat: np.ndarray):
    conn_vertices = population + get_unconnected_vertices(population, adj_mat)
    conn_vertices[conn_vertices.nonzero()] = 1
    unconn_vertices = np.tril(1 - conn_vertices, -1)
    return np.sum(unconn_vertices, axis=1)


if __name__ == '__main__':
    path_to_graph_file = '../Data/jazz.graph'
    adj_matrix = adjacency_matrix(path_to_graph_file)
    # adj_matrix = np.random.choice([0, 1], size=(5, 5), p=[0.4, 0.6])
    graph_dict, _, __ = parse_graph_file(path_to_graph_file)
    print(graph_dict)
    vertex_cover = solve(adj_matrix)
    print("No of vertices in cover are: ", np.sum(vertex_cover))
