import numpy as np

def func(x):
    return 1 - (x-1)*(x-1)

def lipo(n, k, X, f):
    """
        This is the adalipo algorithms described in section 3
        in the work of Malherbe and Vayatis.
        Args:
            n : number of iterations
            K : The specific lipschitz constant, best to use the smallest possible
            X : possible inputs of f
            f : the lipschitz function which to find maxima (as a function object)
        Returns:
            The estimated value which maximizes f.
    """
    # initialization part of the algorithm
    x1 = np.random.choice(X)
    fx = np.zeros((n,)) # holds the functions values computed so far...
    fx[0] = f(x1)
    Xi = [x1]  # holds the functions input values explored so far...
    t = 1
    # Iterations part of the algorithm
    while(t < n):
        Xnew = np.random.choice(X)
        nfx = fx[:t]
        if(np.min(nfx + k * np.linalg.norm(Xnew - Xi)) > np.max(fx)):
            fx[t] = f(Xnew)
            Xi.append(Xnew)
            t += 1
    return Xi[np.argmax(fx)]

def adalipo(n, K, X, f, p):
    """
        This the adalipo algorithm described in section 4
        in the work of Malherbe and Vayatis. It does not require a specific k
        value as lipo, but a grid of values where to search for it.
        Args:
            n : number of iterations
            K : grid of lipschitz constants
            X : possible inputs of f
            f : the lipschitz function which to find maxima (as a function object)
            p : propability of exploitation vs (1-p) propability of exploration
        Returns:
            The estimated value which maximizes f.
    """
    # initialize I as a numpy array of input X
    # for faster computations
    if(len(np.shape(X)) > 1):
        I = np.squeeze(X)
    else:
        I = np.array(X)
    if(len(I.shape) == 1):
        I = np.expand_dims(I, -1)
    # Initialization part of the algorithm
    x1 = np.random.choice(X)
    fx = np.zeros((n,)) # holds the functions values computed so far...
    fx[0] = f(x1)
    Xi = [x1]  # holds the functions input values explored or exploited so far...
    knew = 0
    k = [knew]
    t = 1
    # Iterations part of the algorithm
    while(t < n):
        Bnew = np.random.choice([0, 1], None, [p, 1-p])
        if(Bnew == 1): # exploration
            Xnew = np.random.choice(X)
        else: # exploitation
            mx = np.max(fx, axis=-1)
            if(len(np.shape(X)) > 1):
                Ii = np.squeeze(Xi)
            else:
                Ii = np.array(Xi)
            if(len(Ii.shape) == 1):
                Ii = np.expand_dims(Ii, -1)
            # l contains the norm-2 distance between the points the algorithm
            # has explored and the points in the possible inputs of f
            l = np.zeros((len(Ii), len(I)))
            for idx_i, Iii in enumerate(Ii):
                for idx_j, Ij in enumerate(I):
                    l[idx_i, idx_j] = np.linalg.norm(Iii - Ij)
            # w is an estimation of the term inside the min inside 
            # the Lemma 8 after definition 7
            w = knew * l
            nfx = fx[:t]
            for i in range(np.shape(w)[1]):
                w[:,i] += nfx
            print(w.shape, fx)
            Xkt = X[np.min(w, axis=0) >= mx]
            Xnew = np.random.choice(Xkt)
        Xi.append(Xnew)
        fx[t] = f(Xnew)
        df = np.subtract.outer(fx[:t+1], fx[:t+1])
        fs = []
        for i, xi in enumerate(Xi):
            for j, xj in enumerate(Xi):
                if(xi != xj):
                    fs.append(np.linalg.norm(df[i][j]) / np.linalg.norm(xi-xj))
        knew = np.min(K[K > max(fs)])
        k.append(knew)
        t += 1
    # Output part of the algorithm
    return Xi[np.argmax(fx)]

if __name__ == "__main__":
    X = np.arange(-5, 5, 0.001)
    k = 2
    print(lipo(10, k, X, func))
    K = np.power((1+0.5), np.arange(1, 7, 0.01))
    print(adalipo(20, K, X, func, 0.5))