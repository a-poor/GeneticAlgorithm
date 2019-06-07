# GeneticAlgorithm
by Austin Poor

A Python class to help in running genetic algorithms.

### Args for instantiating the GA:
* `gene_length` => How many possible digits for each gene in a genome
* `genome_length` => How many genes in each genome
* `pop_size` => How many genomes in the population (default=`500`)
* `x_rate` => Percent chance of performing crossover after parents selected (default=`0.9`)
* `mutation_rate` => Percent chance of performing mutation on any given gene in a selected genome (default=`0.005`)
* `parent_selection_method` => Method used for performing parent-selection… (default=`'roulette'`)
    * 'roulette': Choose 2 parents at random where each genome's probability of being chosen is equal to its fitness / total of all fitnesses
    * 'rank': Choose 2 parents at random where each genome's probability of being chosen is equal to its index after sorted by best fitness / the sum of all ranks
    * 'tournament': Randomly select k genomes choose the one with the highest fitness (done twice, to get two parents)
    * 'random': Randomly select 2 genomes at a time
* `k_tournament_select` => Number of genomes in a tournament, when using tournament selection (default=`3`)
* `xover_type` => Type of crossover to use… (default=`'one_point'`)
    * 'one_point': Pick one split point and swap the following two segments
    * 'two_point': Pick two split points and swap the two middle segments
    * 'uniform':  At each gene, "flip a coin" to decide weather or not to crossover genes at that point
* `mutation_type` => One of the following options… (default=`'random_resetting'`)
    * 'random_resetting': Randomly set mutated gene to another number
    * 'swap_mutation': Pick two genes in a genome to swap
    * 'scramble_mutation': Randomly shuffle a subset of the genome
    * 'inversion_mutation': Invert the genome
* `elitism` => What percent of the population (or number of genomes) to preserve unchanged for the next round. (default=`0.05`)
* `random_seed` => Set the random seed (default=`None`)
* `multithread` => Use multithreading for calculating population fitness (default=`False`)

### Args for running the GA:
* `generations` => How many generations to run for (default=`1`)
* `verbose` => Print generation, max_fitness, mean_fitness, fitness_std, and min_fitness as the GA runs? (default=`True`)
* `print_step` => If `verbose`, print status after how many generations? (default=`1`)
* `logfile` => Filename for saving trainint progress – note: will overwrite existing file (default=`None`)
* `stop_value` => Early stopping value, stop trainint early if fitness reaches this threshold (default=`None`)
* `stop_measure` => What metric to watch for early stopping? 'max', 'mean', 'min' (default=`max`)


### Example:
1. Instantiate your genetic algorithm

    `>>> ga = GeneticAlgorithm(2, 30, 500, x_rate=0.9, mutation_rate=0.005, xover_type='two_point')`
2. Redefine the fitness function

    `>>> def fitness(genome):`
    `...    return genome.sum()`
    `>>> ga.fitness = fitness`
3. Run the GA
    
    `>>> ga.run(500, print_step=5, logfile='logtest.csv', stop_value=30, stop_measure='max')`
4. Get the best genomes

    `>>> best_genomes = ga.get_current_population()`
    
    
    
