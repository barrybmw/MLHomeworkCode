#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 23:43:59 2023

@author: liuzhengzuo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define probability distributions
def P_omega1(x, y):
    return 1/(2*np.pi) * np.exp(-(x**2 + y**2)/2)

def P_omega2(x, y):
    return 1/(2*np.pi) * np.exp(-((x-2)**2 + (y-2)**2)/2)

# Generate 1000 samples
samples = []
for i in range(1000):
    # Sample random variable
    rv = np.random.choice([0, 1], p=[2/3, 1/3])

    # Generate (x, y) based on the random variable
    if rv == 0:
        x = np.random.normal(0, 1, size=2)
        while P_omega1(*x) == 0: # Rejection sampling to ensure non-zero density
            x = np.random.normal(0, 1, size=2)
    else:
        x = np.random.normal([2, 2], 1, size=2)
        while P_omega2(*x) == 0: # Rejection sampling to ensure non-zero density
            x = np.random.normal([2, 2], 1, size=2)
            
    samples.append(x)

# Convert samples to NumPy array and print shape
samples = np.array(samples)

# Use Parzen window of proper width to estimate the density function of P(x,y)
x, y = np.mgrid[-5:5:0.02, -5:5:0.02]
pos = np.dstack((x, y))
window_width = 0.5
parzen_pdf = np.zeros_like(x)
for i in range(len(samples)):
    parzen_pdf += multivariate_normal.pdf(pos, mean=samples[i], cov=window_width**2 * np.eye(2))
parzen_pdf /= len(samples)

# Plot the two-dimensional density as heatmap
plt.imshow(parzen_pdf.T, origin='lower', extent=[-5, 5, -5, 5], cmap='plasma')
plt.colorbar()
plt.show()

