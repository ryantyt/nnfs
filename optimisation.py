import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
import numpy as np
from object import Layer_Dense, Activation_ReLU, Activation_Softmax, Loss, Loss_CategoricalCrossEntropy

nnfs.init()

X, y = vertical_data(samples=100, classes=3) 

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_ReLU()

dense3 = Layer_Dense(3, 3)
activation3 = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()

# Helper variables to track best loss, weights, and biases
lowest_loss = 99999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
best_dense3_weights = dense3.weights.copy()
best_dense3_biases = dense3.biases.copy()


# Repeat many times with random numbers to find the combination with the lowest loss
for iteration in range(10000):

    # Set of weights and biases
    n1 = 0.05
    dense1.weights += n1 * np.random.randn(2, 3)
    dense1.biases += n1 * np.random.randn(1, 3)
    dense2.weights += n1 * np.random.randn(3, 3)
    dense2.biases += n1 * np.random.randn(1, 3)
    dense3.weights += n1 * np.random.randn(1, 3)
    dense3.biases += n1 * np.random.randn(1, 3)

    # Make a forward pass with this set
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)


    # Returns the loss on the second activation layer
    loss = loss_function.calculate(activation3.output, y)
    
    # Calculate accuracy from output of activation2 and targets
    # Calculate values along first axis
    predictions = np.argmax(activation3.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print('New set of weights found, iteration: ', iteration, '\nLoss: ', loss, '\nAccuracy: ', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        best_dense3_weights = dense3.weights.copy()
        best_dense3_biases = dense3.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
        dense3.weights = best_dense3_weights.copy()
        dense3.biases = best_dense3_biases.copy()


# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg') 
