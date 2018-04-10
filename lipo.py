import numpy as np

def func(x):
    return 1 - (x-1)*(x-1)

def lipo(n, k, X, f):
    x1 = np.random.choice(X)
    fx = [f(x1)]
    Xi = [x1]
    t = 1
    while(t < n):
        Xnew = np.random.choice(X)
        if(np.min(fx + k * np.linalg.norm(Xnew - Xi)) > np.max(fx)):
            fx.append(f(Xnew))
            Xi.append(Xnew)
            t = t + 1
    return Xi[np.argmax(fx)]

def adalipo(n, K, X, f, p):
    x1 = np.random.choice(X)
    fx = [f(x1)]
    Xi = [x1]
    knew = 0
    k = [knew]
    t = 1
    while(t < n):
        Bnew = np.random.choice([0, 1], None, [p, 1-p])
        if(Bnew == 1):
            Xnew = np.random.choice(X)
        else:
            mx = np.max(fx, axis=-1)
            l = np.squeeze(np.subtract.outer(Xi, X))
            print(np.shape(l))
            s1 = l[:,::2,:,::2]
            s2 = l[:,1::2,:,1::2]
            w = np.reshape(np.concatenate(s1, s2).squeeze(), (-1, np.shape(X)[-1]))
            Xkt = X[np.min(fx + knew * np.linalg.norm(w, axis=1)) >= mx]
            Xnew = np.random.choice(Xkt)
        Xi.append(Xnew)
        fx.append(f(Xnew))
        df = np.subtract.outer(fx, fx)
        fs = []
        for i, xi in enumerate(Xi):
            for j, xj in enumerate(Xi):
                print i,j,": ", df[i][j]
                if(xi != xj):
                    fs.append(np.linalg.norm(df[i][j]) / np.linalg.norm(xi-xj))
        knew = min(K[K > max(fs)])
        k.append(knew)
        t = t + 1
    return Xi[np.argmax(fx)]

if __name__ == "__main__":
    X = np.arange(-5, 5, 0.001)
    k = 2
    # print(lipo(10, k, X, func))
    K = np.power((1+0.5), np.arange(1, 7, 0.01))
    print(adalipo(10, K, X, func, 0.5))