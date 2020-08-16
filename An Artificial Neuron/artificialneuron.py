
import math
import numpy as np

def relu(x):
   return np.maximum(0,x)


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


def activate(inputs, weights, activation_func):
    h = 0
    for x, w in zip(inputs, weights):
        h += x*w
        
    if activation_func == "relu":
        return relu(h)
    elif activation_func == "sigmoid":
        return sigmoid(h)


if __name__ == "__main__":
    inputs = [0.509, 0.2, 0.1]
    weights = [0.1, 0.997, 0.1]
    output = activate(inputs, weights, activation_func="sigmoid")
    print(output)