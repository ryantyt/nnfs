import numpy as np

softmax_output = np.array([[0.13, 0.1, 0.51], 
                            [0.77, 0.36, 0.71], 
                            [0.63, 0.95, 0.18]])

class_targets = [0, 1, 1]

print(softmax_output[[0, 1, 2], class_targets])