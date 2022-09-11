import numpy as np

# Gradient used as example is a vector of 1's
dvalues = np.array([[1., 1., 1.],
                      [2., 2., 2.],
                      [3., 3., 3.]])


# 3 sets of weights - 1/neuron
# 4 inputs, 4 weights
# Weights are kept transposed so that coordinates match for each layer (remember matrix multiplication)
weights = np.array([[0.2, 0.8, -0.5, 1],
                      [0.5, -0.91, 0.26, -0.5],
                      [-0.26, -0.27, 0.17, 0.87]]).T

# Summing dvalues * weights.T then putting it into a numpy array can be 
# replaced using dot function of np

dinputs = np.dot(dvalues, weights.T)

print(dinputs)