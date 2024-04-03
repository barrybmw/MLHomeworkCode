#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 18:15:28 2023

@author: liuzhengzuo
"""

import numpy as np
from hmmlearn import hmm

# Load data
sequences = np.load("/Users/liuzhengzuo/Desktop/PRML/homework6/sequences.npy")-1
sequences_train = sequences[0:150,:]
sequences_validate = sequences[150:,:]

# Prepare data for HMM
X_train = sequences_train.reshape(-1, 1)
X_validate = sequences_validate.reshape(-1, 1)

best_score = best_model = None
n_fits = 200
np.random.seed(13)
for idx in range(n_fits):
    model = hmm.CategoricalHMM(
        n_components=2, random_state=idx,
        init_params='se')  # don't init transition, set it below
    # we need to initialize with random transition matrix probabilities
    # because the default is an even likelihood transition
    # we know transitions are rare (otherwise the casino would get caught!)
    # so let's have an Dirichlet random prior with an alpha value of
    # (0.1, 0.9) to enforce our assumption transitions happen roughly 10%
    # of the time
    model.transmat_ = np.array([np.random.dirichlet([0.9, 0.1]),
                                np.random.dirichlet([0.1, 0.9])])
    model.fit(X_train)
    score = model.score(X_validate)
    #print(f'Model #{idx}\tScore: {score}')
    if best_score is None or score > best_score:
        best_model = model
        best_score = score

print(f'Initial Probabilites:\n{best_model.startprob_.round(3)}\n\n')
print(f'Transmission Probabilites:\n{best_model.transmat_.round(3)}\n\n')
print(f'Emission Probabilites:\n{best_model.emissionprob_.round(3)}\n\n')

sequence1 = np.array([3, 2, 1, 3, 4, 5, 6, 3, 1, 4, 1, 6, 6, 2, 6])
X1 = sequence1.reshape(-1, 1)-1


model1 = hmm.CategoricalHMM(n_components=2, params='s', init_params='s')
model1.transmat_ = best_model.transmat_
model1.emissionprob_ = best_model.emissionprob_
model1.fit(X1)

Z = model1.predict(X1)
print(Z)

print(f'Initial Probabilites:\n{model1.startprob_.round(3)}\n\n')
print(f'Transmission Probabilites:\n{model1.transmat_.round(3)}\n\n')
print(f'Emission Probabilites:\n{model1.emissionprob_.round(3)}\n\n')
