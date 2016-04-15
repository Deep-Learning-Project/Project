import bottleneck as bn
import numpy as np
import theano
from tqdm import trange

from DataLoader import load_data
from NetworkBuilder import build_network

pop_size = 50
num_gen = 400
xover_rate = 0.7
mut_rate = 0.01
num_elite = 3


def solve(X_train, y_train, X_test, y_test):
    """
    The basic genetic algorithm. Graph is represented by an adjacency matrix
    :param adj_mat: Adjacency matrix of graph instance required to solve
    :return: The best found vertex cover for the graph.
    """
    # Generate pop_size number of random solutions. Solutions may be infeasible
    population = generate_population(pop_size, 136)
    for generation in trange(num_gen):
        # Calculate the "goodness" of each solution and give it a score
        fitness_scores = evaluate_fitness(population, X_train, y_train, X_test, y_test)
        # Introduce concept of elitism. In each generation, num_elite number of best solutions will be chosen to be
        # carried forward to the next iteration without any modifications to them. Instead of sorting entire array,
        # sort it partially so that only top num_elite number of solutions are sorted, and the rest remains same
        if num_elite != 0:
            part_sorted = bn.argpartsort(fitness_scores, fitness_scores.shape[0] - num_elite)
            elite_indices = part_sorted[-num_elite:]  # Sorted top num_elite no of solutions
            remaining_indices = part_sorted[:-num_elite]  # Unsorted solutions
        # Print the best fitness score of every 20 generations to see how algorithm is performing.
        if generation % 10 == 0:
            print(np.max(fitness_scores))
        # Select parents which will create the next generation
        if num_elite != 0:
            parents = select_parents(population[remaining_indices], fitness_scores[remaining_indices])
        else:
            parents = select_parents(population, fitness_scores)
        # Perform a crossover operation on selected parents
        children = crossover(parents, xover_rate)
        # Perform a mutation operation on crossovered parents
        mutate(children, mut_rate)

        if num_elite != 0:
            # Add the elite solutions that were removed from parents, and set them as population (parents) for next gen
            population = np.vstack((children, population[elite_indices]))
        else:
            population = children

    # Find solution with highest fitness scores, i.e. solution with least number of vertices.
    best_solution_index = np.argmax(fitness_scores)
    return population[best_solution_index]


def generate_population(population_size, num_bits):
    """
    Generates population_size number of random solutions for the vertex cover. May generate invalid solutions.
    :param population_size: Number of solutions to be generated
    :param num_bits: Size of graph instance. Required for binary representation of vertex cover
    :return: A 2D ndarray of shape (population_size, num_bits) where each row indicates a solution to the graph
    instance
    """
    population = np.random.random_integers(0, 1, size=(population_size, num_bits))
    return population


def evaluate_fitness(population: np.ndarray, X_train, y_train, X_test, y_test, batch_size=5000, nb_epoch=20):
    # Return the number of vertices not in cover
    ret_val = np.zeros(population.shape[0])
    for ind, bit_array in enumerate(population):
        try:
            theano.gof.cc.get_module_cache().clear()
            model = build_network(bit_array, input_shape=(3, 32, 32), n_classes=10)
            print(model.summary())
            model.compile(optimizer='adadelta', loss='categorical_crossentropy')
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)
            ret_val[ind] = model.evaluate(X_test, y_test, batch_size=1000, show_accuracy=True)[1]
            print(ret_val[ind])
        except Exception as e:
            print("Skipping model...\nException is:", e)
    print(ret_val)
    return ret_val


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


def crossover(parents: np.ndarray, crossover_rate: float = xover_rate, xover_type='uox'):
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


def mutate(children: np.ndarray, mutation_rate):
    for row in range(0, children.shape[0], 2):
        if np.random.binomial(1, mutation_rate):
            mut_pos = np.random.random_integers(1, children.shape[1] - 1)
            children[row, mut_pos] = 1 - children[row, mut_pos]


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_data(red_size=0.1)
    best_model = solve(X_train, y_train, X_test, y_test)
    model = build_network(best_model)
    print(model.summary())
    model.compile(optimizer='adadelta', loss='categorical_crossentropy')
    model.fit(X_train, y_train, batch_size=2048, nb_epoch=20)
    print(model.evaluate(X_test, y_test))
