# Log solves for x in the equation of e^x = b
# Categorical cross entropy = negative log of predicted class value

import numpy as np
import math

softmax_output = [0.3, 0.9, 0.8]
target_output = [1, 0, 0]

#if target value's 0 then

loss = -(math.log(softmax_output[0])*target_output[0]+
        math.log(softmax_output[1])*target_output[1] +
        math.log(softmax_output[2])*target_output[2])

print(loss)