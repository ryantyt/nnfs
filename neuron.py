import numpy as np
import random

# Making a neuron from https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=TLPQMTUwODIwMjIwRbO4sY7FsA&index=2&ab_channel=sentdex
# Every neuron has a connection to every single previous neuron
inputs = [[-2.2, 2.3, 2.9, 4.9],
            [-3.4, -2.8, -2.8, 3.8], 
            [0.7, -2.2, 0.3, 0.5]]

weights = [[-0.5, 0.1, 0., 1.0],
            [3.03, 2.97, 2.97, 4.11], 
            [-4.96, -0.83, 0.43, 4.46]]

# Add up all the inputs * weights add bias
biases = [2, 3, 0.5]

weights2 = [[-2.5, 3.6, 0.4],
            [-2.7, 3.0, -1.7],
            [-3.8, -4.6, -2.6]]

biases2 = [0.4, 1.0, 5.0]

# layer_outputs = [] # Output of current layer
# for neuron_weights, neuron_biases in zip(weights, biases):
#     neuron_output = 0 # Output of a given neuron
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += n_input*weight # 
#     neuron_output += neuron_biases # Add bias
#     layer_outputs.append(neuron_output)

# print(layer_outputs)

layer1_output = np.dot(inputs, np.array(weights).T) + biases
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
print(layer2_output)
