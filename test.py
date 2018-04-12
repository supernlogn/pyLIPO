from lipo import *
import numpy as np

def func(x):
    return 1 - (x-1)*(x-1)

if __name__ == "__main__":
    X = np.arange(-5, 5, 0.001)
    k = 7
    print(lipo(10, k, X, func))
    K = np.power((1+0.5), np.arange(1, 7, 0.01))
    print(adalipo(20, K, X, func, 0.5))