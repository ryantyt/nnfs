from cmath import exp
# Normalisation: each value in an output layer divided by the sum of all the values in the output layer
# If u = [1, 2] and y = normalised u then y = 1/3, 2/3
# Provides a probability distribution
# Exponentiating with e beforehand to get rid of negative values and maintain the ratio

import math
import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],
                 [0.8, 2.89, -0.1],
                 [-8.278, 4.29, -5.313]]

exp_values = np.exp(layer_outputs)

norm_values = exp_values / np.sum(layer_outputs, axis=1, keepdims=True)

print(norm_values)
# print(sum(norm_values))

# To prevent overflow subtract everything in the inputs by the largest value in the array
