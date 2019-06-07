#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Austin
"""

import numpy as np
import threading


class GeneticAlgorithm:
    """
    Class to help in testing Genetic Algorithms
    
    Args:
    gene_length => How many possible digits for each gene in a genome
    genome_length => How many genes in each genome
    pop_size => How many genomes in the population (default=500)
    x_rate => Percent chance of performing crossover after parents selected (default=0.9)
    mutation_rate => Percent chance of performing mutation on any given gene in a selected genome (default=0.005)
    parent_selection_method => Method used for performing parent-selection… (default='roulette')
        'roulette': Choose 2 parents at random where each genome's probability of being chosen is equal to its fitness / total of all fitnesses
        'rank': Choose 2 parents at random where each genome's probability of being chosen is equal to its index after sorted by best fitness / the sum of all ranks
        'tournament': Randomly select k genomes choose the one with the highest fitness (done twice, to get two parents)
        'random': Randomly select 2 genomes at a time
    k_tournament_select => Number of genomes in a tournament, when using tournament selection (default=3)
    xover_type => Type of crossover to use… (default='one_point')
        'one_point': Pick one split point and swap the following two segments
        'two_point': Pick two split points and swap the two middle segments
        'uniform':  At each gene, "flip a coin" to decide weather or not to crossover genes at that point
    mutation_type => One of the following options… (default='random_resetting')
        'random_resetting': Randomly set mutated gene to another number
        'swap_mutation': Pick two genes in a genome to swap
        'scramble_mutation': Randomly shuffle a subset of the genome
        'inversion_mutation': Invert the genome
    elitism => What percent of the population (or number of genomes) to preserve unchanged for the next round. (default=0.05)
    random_seed => Set the random seed (default=None)
    multithread => Use multithreading for calculating population fitness (default=False)
    
    
    Example:
    1. Instantiate your genetic algorithm
        >>> ga = GeneticAlgorithm(2, 30, 500, x_rate=0.9, mutation_rate=0.005, xover_type='two_point')
    2. Redefine the fitness function
        >>> def fitness(genome):
        ...    return genome.sum()
        >>> ga.fitness = fitness
    3. Run the GA
        >>> ga.run(500, print_step=5, logfile='logtest.csv', stop_value=30, stop_measure='max')
    4. Get the best genomes
        >>> best_genomes = ga.get_current_population()
    
    """
    
    def __init__(self, gene_length, genome_length, pop_size=500, x_rate=0.9, mutation_rate=0.005, parent_selection_method='roulette', k_tournament_select=3, xover_type='one_point', mutation_type='random_resetting', elitism=0.05, random_seed=None, multithread=False):
        # Settings
        self.gene_length = gene_length
        self.genome_length = genome_length
        self.pop_size = pop_size
        self.x_rate = x_rate
        self.mutation_rate = mutation_rate
        if 0 < elitism < 1:
            self.elitism = int(np.round(elitism * pop_size))
        else:
            self.elitism = elitism
        
        self.population = None
        self.pop_fitness = None
        self.generation = 0
        self.max_fitness = None
        self.avg_fitness = None
        self.std_fitness = None
        self.min_fitness = None
        
        np.random.seed(random_seed)
        
        self.multithread = multithread
        
        # Set crossover method
        assert xover_type in ['one_point', 'two_point', 'uniform'],\
        "xover_type must be one of the following: 'one_point', 'two_point', 'uniform'"
        if xover_type == 'one_point':
            self.crossover = self.one_point_xover
        elif xover_type == 'two_point':
            self.crossover = self.two_point_xover
        elif xover_type == 'uniform':
            self.crossover = self.uniform_xover
        self.xover_type = xover_type
        
        # Set method of parent selection
        assert parent_selection_method in ['roulette', 'rank', 'tournament', 'random'],\
        "selection_type must be one of the following: 'roulette', 'rank', 'tournament', 'random'"
        if parent_selection_method == 'roulette':
            self.parent_selection = self.roulette_selection
        elif parent_selection_method == 'rank':
            self.parent_selection = self.rank_selection
        elif parent_selection_method == 'tournament':
            self.parent_selection = self.tournament_selection
        elif parent_selection_method == 'random':
            self.parent_selection = self.random_selection
        self.parent_selection_method = parent_selection_method
        self.k_tournament_select = k_tournament_select
        
        # Set mutation method
        assert mutation_type in ['random_resetting', 'swap_mutation', 'scramble_mutation', 'inversion_mutation'], \
        "mutation_type must be one of the following: 'random_resetting', 'swap_mutation', 'scramble_mutation', 'inversion_mutation'"
        if mutation_type == 'random_resetting':
            self.mutate = self.random_mutation
        elif mutation_type == 'swap_mutation':
            self.mutate = self.swap_mutation
        elif mutation_type == 'scramble_mutation':
            self.mutate = self.scramble_mutation
        elif mutation_type == 'inversion_mutation':
            self.mutate = self.inversion_mutation
        self.mutation_type = mutation_type
        
        return
    
    def random_genome(self):
        return np.random.randint(self.gene_length, size=(self.genome_length,))
    
    def make_population(self):
        return np.array([self.random_genome() for _ in range(self.pop_size)])
    
    def fitness(self, genome):
        """Replace this function with a user-defined fitness function."""
        raise FitnessUndefinedError
        return
    
    def place_fitness(self, a, i, genome):
        a[i] = self.fitness(genome)
        return
    
    def evaluate_pop_fitness(self):
        try:
            assert self.population is not None
        except:
            raise NoPopulationError
            
        if self.multithread:
            fitnesses = np.zeros(len(self.population))
            threads = []
            for i, g in enumerate(self.population):
                threads.append(threading.Thread(target=self.place_fitness, args=(fitnesses, i, g)))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            fitnesses = np.array([self.fitness(g) for g in self.population])
        
        # Set current fitness metrics
        self.max_fitness = fitnesses.max()
        self.avg_fitness = fitnesses.mean()
        self.std_fitness = fitnesses.std()
        self.min_fitness = fitnesses.min()
        self.pop_fitness = fitnesses
        return 
    
    def one_point_xover(self, g1, g2):
        split = np.random.randint(1, self.genome_length-1)
        new_g1 = np.concatenate((g1[:split], g2[split:]))
        new_g2 = np.concatenate((g2[:split], g1[split:]))
        return new_g1, new_g2
    
    def two_point_xover(self, g1, g2):
        start = np.random.randint(0, self.genome_length-1)
        end = np.random.randint(start, self.genome_length)
        new_g1 = np.concatenate((g1[:start], g2[start:end], g1[end:]))
        new_g2 = np.concatenate((g2[:start], g1[start:end], g2[end:]))
        return new_g1, new_g2
    
    def uniform_xover(self, g1, g2):
        new_g1 = []
        new_g2 = []
        for a, b in zip(g1, g2):
            if np.random.random() >= 0.5:
                new_g1.append(a)
                new_g2.append(b)
            else:
                new_g1.append(b)
                new_g2.append(a)
        return np.array(new_g1), np.array(new_g2)
    
    def crossover(self, g1, g2):
        raise UndefinedXoverError
        return
    
    def set_crossover(self, xover_type):
        assert xover_type in ['one_point', 'two_point', 'uniform'],\
        "xover_type must be one of the following: 'one_point', 'two_point', 'uniform'"
        if xover_type == 'one_point':
            self.crossover = self.one_point_xover
        elif xover_type == 'two_point':
            self.crossover = self.two_point_xover
        elif xover_type == 'uniform':
            self.crossover = self.uniform_xover
        self.xover_type = xover_type
        return
        
    def random_mutation(self, genome):
        for i, n in enumerate(genome):
            if np.random.random() < self.mutation_rate:
                while genome[i] == n:
                    genome[i] = np.random.randint(self.gene_length)
        return genome
    
    def swap_mutation(self, genome):
        p1 = np.random.randint(self.gene_length)
        p2 = np.random.randint(self.gene_length)
        while p2 == p1:
            p2 = np.random.randint(self.gene_length)
        new_genome = genome.copy()
        new_genome[p1] = genome[p2]
        new_genome[p2] = genome[p1]
        return new_genome
    
    def scramble_mutation(self, genome):
        p1 = np.random.randint(self.gene_length-1)
        p2 = np.random.randint(p1, self.gene_length)
        new_genome = genome.copy()
        new_genome[p1:p2] = genome[p1:p2][::-1]
        return new_genome
    
    def inversion_mutation(self, genome):
        return genome[::-1]
    
    def mutate(self, genome):
        raise UndefinedMutationError
        return
    
    def set_mutation(self, mutation_type):
        assert mutation_type in ['random_resetting', 'swap_mutation', 'scramble_mutation', 'inversion_mutation'], \
        "mutation_type must be one of the following: 'random_resetting', 'swap_mutation', 'scramble_mutation', 'inversion_mutation'"
        if mutation_type == 'roulette':
            self.mutate = self.roulette_selection
        elif mutation_type == 'rank':
            self.mutate = self.rank_selection
        elif mutation_type == 'tournament':
            self.mutate = self.tournament_selection
        elif mutation_type == 'random':
            self.mutate = self.random_selection
        self.mutation_type = mutation_type
        return
    
    def roulette_selection(self):
        pop_fitnesses = self.pop_fitness.copy()
        if pop_fitnesses.min() < 0:
            pop_fitnesses = pop_fitnesses + pop_fitnesses.min()
        pop_fitnesses = pop_fitnesses / pop_fitnesses.sum()
        return self.population[np.random.choice(np.arange(self.pop_size), size=2, p=pop_fitnesses)]
    
    def rank_selection(self):
        fitnesses = self.pop_fitness.copy()
        index_order = fitnesses.argsort()
        prob = index_order / index_order.sum()
        return self.population[np.random.choice(np.arange(self.pop_size), size=2, p=prob)]
    
    def tournament_selection(self):
        subpop = self.population[np.random.randint(0, self.pop_size, self.k_tournament_select)]
        subpop_fitnesses = np.array([self.fitness(g) for g in subpop])
        g1 = subpop[subpop_fitnesses.argmax()]
        subpop = self.population[np.random.randint(0, self.pop_size, self.k_tournament_select)]
        subpop_fitnesses = np.array([self.fitness(g) for g in subpop])
        g2 = subpop[subpop_fitnesses.argmax()]
        return np.array([g1, g2])
    
    def random_selection(self):
        return self.population[np.random.randint(0, self.pop_size, 2)]
    
    def parent_selection(self):
        raise UndefinedSelectionError
        return
    
    def set_selection(self, parent_selection, k_tournament_select=3):
        assert parent_selection in ['roulette', 'rank', 'tournament', 'random'],\
        "selection_type must be one of the following: 'roulette', 'rank', 'tournament', 'random'"
        if parent_selection == 'roulette':
            self.parent_selection = self.roulette_selection
        elif parent_selection == 'rank':
            self.parent_selection = self.rank_selection
        elif parent_selection == 'tournament':
            self.parent_selection = self.tournament_selection
        elif parent_selection == 'random':
            self.parent_selection = self.random_selection
        self.parent_selection = parent_selection
        self.k_tournament_select = k_tournament_select
        return
        
    def save_population(self, filename, filetype='npz'):
        assert filetype in ['npy', 'txt', 'npz'], "filetype should be one of the following: 'npy', 'txt', 'npz'"
        if filetype == 'npy':
            np.save(filename, self.population)
        elif filetype == 'txt':
            np.savetxt(filename, self.population)
        elif filetype == 'npz':
            np.savez(filename, self.population)
        return
    
    def get_current_population(self):
        self.evaluate_pop_fitness()
        return self.population[self.pop_fitness.argsort()][::-1]
    
    def get_elites(self):
        fitnesses = self.pop_fitness.copy()
        indexes = fitnesses.argsort()
        return self.population[indexes[-self.elitism:]]
    
    def run(self, generations=1, verbose=True, print_step=1, logfile=None, stop_value=None, stop_measure='max'):
        if logfile is not None:
            log = ['generation,max_fitness,mean_fitness,fitness_std,min_fitness']
        if self.population is None:
            self.population = self.make_population()
        if verbose:
            print('generation   max_fitness   mean_fitness   fitness_std   min_fitness')
        for n_gen in range(generations):
            new_generation = np.zeros((1,self.genome_length), dtype='int')
            self.evaluate_pop_fitness()
            if self.elitism > 0:
                new_generation = np.concatenate((new_generation, self.get_elites()))
            while len(new_generation) < self.pop_size + 1:
                g1, g2 = self.parent_selection()
                if np.random.random() < self.x_rate:
                    g1, g2 = self.crossover(g1, g2)
                g1 = self.mutate(g1)
                g2 = self.mutate(g2)
                new_generation = np.concatenate((new_generation, np.array([g1,g2])))
            self.population = new_generation[1:self.pop_size+1]
            if verbose and (n_gen % print_step == 0):
                print(' %7i      %8.2f       %8.2f      %8.2f      %8.2f' % (n_gen, self.max_fitness, self.avg_fitness, self.std_fitness, self.min_fitness))
            if logfile is not None:
                log.append('%s,%s,%s,%s,%s' % (n_gen, self.max_fitness, self.avg_fitness, self.std_fitness, self.min_fitness))
            if stop_value is not None:
                if stop_measure == 'max':
                    if self.max_fitness >= stop_value:
                        print('--- STOP VALUE REACHED ---')
                        break
                elif stop_measure == 'mean' or stop_measure == 'avg':
                    if self.avg_fitness >= stop_value:
                        print('--- STOP VALUE REACHED ---')
                        break
                elif stop_measure == 'min':
                    if self.min_fitness >= stop_value:
                        print('--- STOP VALUE REACHED ---')
                        break
        if verbose:
            print('\nFINAL POPULATION FITNESS:')
            print(' %7i      %8.2f       %8.2f      %8.2f      %8.2f\n\n' % (n_gen, self.max_fitness, self.avg_fitness, self.std_fitness, self.min_fitness))
        if logfile is not None:
            print('Saving logfile.')
            with open(logfile, 'w') as f:
                f.write('\n'.join(log))    
            print('logfile saved.')
        return

    
# Define some custom Exceptions
class FitnessUndefinedError(Exception):
    def __str__(self):
        return "Fitness function left undefined. Please redefine it before running."
    
class NoPopulationError(Exception):
    def __str__(self):
        return "No current population."
    
class UndefinedXoverError(Exception):
    def __str__(self):
        return "Set a method of crossover with: GeneticAlgorithm.set_crossover(xover_type)"
    
class UndefinedSelectionError(Exception):
    def __str__(self):
        return "Set a method of parent selection with: GeneticAlgorithm.set_selection(parent_selection, k_tournament_select=3)"
    
class UndefinedMutationError(Exception):
    def __str__(self):
        return "Set a method of mutation with: GeneticAlgorithm.set_mutation(mutation_type)"

