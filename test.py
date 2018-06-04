from lipo import *
import numpy as np

def func(x):
    return 1 - (x-1)*(x-1)

def func2(x):
  return 1 -(x[0] - 2)*(x[0] - 2)*(x[1] - 1)*(x[1] - 1)


if __name__ == "__main__":
    X = np.arange(-5, 5, 0.001)
    k = 7
    print(lipo(10, k, X, func))
    K = np.power((1+0.5), np.arange(1, 7, 0.01))
    print(adalipo(20, K, np.expand_dims(X, 1), func, 0.5))
    # test 2D
    X = np.reshape(np.stack(np.meshgrid(np.arange(-10,10,0.01), np.arange(-10,10,0.01)), axis=-1), [-1,2])
    K = np.concatenate([np.arange(0.1,1.5, 0.01), np.power((1+0.5), np.arange(2.0, 49.0, 0.1))])
    print(adalipo(20, K, X, func2, 0.5))
