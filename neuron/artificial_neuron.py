import math

def sigmond(x):
    y = 1.0 / (1 + math.exp(-x))
    return y

def activate(inputs, weights):
    # perform net input
    h = 0
    for x, w in zip(inputs, weights):
        h += x * w

    # perform activation
    return sigmond(h)


inputs = [0.5, 0.3, 0.2]
weights = [0.4, 0.7, 0.2]
output = activate(inputs, weights)
print(output)
