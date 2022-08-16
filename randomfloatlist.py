import random

a, b, c = 0, 1, 2


print([float(f"%.{c}f" % random.uniform(a, b)) for _ in range(3)])