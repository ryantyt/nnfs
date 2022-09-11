# Backprop via chain rule

x = [1.0, -2.0, 3.0]  # input values
w = [-3.0, -1.0, 2.0]  # weights
b = 1.0  # bias

# # ReLU(∑inputs*weights + biases) = big sum log equation
# # y = ReLU(sum(mul(x0, w0), mul(x1, w1), mul(x2, w2), b))
# # 3 nested functions: ReLU, sum of weighted inputs and biases, multiplication of inputs and weights

# # Backwards pass:
# Multiplying inputs by weights
xw0 = x[0]*w[0]
xw1 = x[1]*w[1]
xw2 = x[2]*w[2]

z = xw0 + xw1 + xw2 + b

y = max(z, 0)

# Derivative from next layer
dvalue = 1.0

# Derivation of ReLU and then chain rule
drelu_dz = dvalue * (1. if z > 0 else 0.)

# Partial derivatives of the mutliplication then chain rule
dsum_dxw0 = 1
drelu_dxw0 = drelu_dz * dsum_dxw0

dsum_dxw1 = 1
drelu_dxw1 = drelu_dz * dsum_dxw1

dsum_dxw2 = 1
drelu_dxw2 = drelu_dz * dsum_dxw2

dsum_db = 1
drelu_db = drelu_dz * dsum_db

print(drelu_db, drelu_dxw0, drelu_dxw1, drelu_dxw2)

# Partial derivatives then chain rule
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2
print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

dx = [drelu_dx0, drelu_dx1, drelu_dx2]  # gradients on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2]  # gradients on weights
db = drelu_db  # gradient on bias...just 1 bias here

n = -0.001
w[0] += n * dw[0]
w[1] += n * dw[1]
w[2] += n * dw[2]
b += n * db

# Multiplying inputs by weights
xw0 = x[0]*w[0]
xw1 = x[1]*w[1]
xw2 = x[2]*w[2]

z = xw0 + xw1 + xw2 + b

y = max(z, 0)
print(y)