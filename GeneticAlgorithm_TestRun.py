#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: a-poor

Sample Genetic Algorithm
"""

import numpy as np
from GenAlgLib import GeneticAlgorithm


# Defining custom fitness function
def fitness(genome):
    total = 1
    for n in genome:
        total += np.sqrt(np.e**n)
    return total

# Create instance of GA class
ga = GeneticAlgorithm(10, 30, 500, x_rate=0.9, mutation_rate=0.005, multithread=True, random_seed=0)
# Re-define object's fitness function to the one I created
ga.fitness = fitness
# Run the GA for 500 generations
ga.run(500, print_step=5, logfile='logtest.csv', stop_value=2500, stop_measure='max')
