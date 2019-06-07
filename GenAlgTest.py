#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:51:42 2019

@author: Austin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GenAlgLib import GeneticAlgorithm


def fitness(genome):
    total = 1
    for n in genome:
        total += np.e**n
    return total

ga = GeneticAlgorithm(10, 30, 350, multithread=False, random_seed=0)
ga.fitness = fitness
ga.run(500, print_step=1, logfile='logs/logtest_3.csv')


df = pd.read_csv('logs/logtest_2.csv')

plt.plot(df.max_fitness)
plt.plot(df.mean_fitness)
plt.plot(df.min_fitness)
plt.xlabel('generation')
plt.ylabel('fitness')
plt.grid()
plt.legend()
plt.plot()