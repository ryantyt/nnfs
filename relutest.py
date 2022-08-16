import numpy as np

np.random.seed(0)
# ReLU is just making negative values zero and anything greater than zero stays the same

X = [[-2.2, 2.3, 2.9, 4.9],
    [-3.4, -2.8, -2.8, 3.8], 
    [0.7, -2.2, 0.3, 0.5]]

inputs = [-58.1, -40.6, 50.5, -75.8, -90.7, -35.1, -65.2]
output = []

for i in inputs:
    output.append(max(0, i))

print(output)
