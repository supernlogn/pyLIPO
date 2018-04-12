import numpy as np

__all__ = ["lipo", "adalipo"]

def lipo(n, k, X, f, MAX_ITERATIONS_PER_T = 100):
  """
    This is the adalipo algorithms described in section 3
    in the work of Malherbe and Vayatis.
    Args:
      n : number of iterations
      K : The specific lipschitz constant, best to use the smallest possible
      X : possible inputs of f
      f : the lipschitz function which to find maxima (as a function object)
      MAX_ITERATIONS_PER_T : (optional) max iterations to do per timestep t, the algorithm can find itself
                              in an endless loop cause it depends to propability in order to go to the next timestep.  
    Returns:
      The estimated value which maximizes f.
  """
  # initialization part of the algorithm
  x1 = np.random.choice(X)
  fx = np.zeros((n,)) # holds the functions values computed so far...
  fx[0] = f(x1)
  Xi = [x1]  # holds the functions input values explored so far...
  t = 1
  it_per_t = 0
  # Iterations part of the algorithm
  while(t < n and it_per_t < MAX_ITERATIONS_PER_T):
    Xnew = np.random.choice(X)
    nfx = fx[:t]
    it_per_t += 1
    if(np.min(nfx + k * np.linalg.norm(Xnew - Xi)) > np.max(fx)):
      fx[t] = f(Xnew)
      Xi.append(Xnew)
      t += 1
      it_per_t = 0
  return Xi[np.argmax(fx)]

def adalipo(n, K, X, f, p):
  """
      This the adalipo algorithm described in section 4
      in the work of Malherbe and Vayatis. It does not require a specific k
      value as lipo, but a grid of values where to search for it.
      Args:
          n : number of iterations
          K : grid of lipschitz constants (non-decreasing list or numpy array)
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
  # make sure that K will be a numpy array
  Kg = np.array(K)
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
      Xkt = X[np.min(w, axis=0) >= mx]
      if(len(Xkt) != 0): # exploit
        Xnew = np.random.choice(Xkt)
      else: # explore again...
        Xnew = np.random.choice(X)
    Xi.append(Xnew)
    fx[t] = f(Xnew)
    df = np.subtract.outer(fx[:t+1], fx[:t+1])
    fs = np.zeros(((len(Xi)) * (len(Xi) - 1), ))
    idx = 0
    for i, xi in enumerate(Xi):
      for j, xj in enumerate(Xi):
        if(xi != xj):
          fs[idx] = np.linalg.norm(df[i][j]) / np.linalg.norm(xi-xj)
          idx += 1
    knew = np.min(Kg[Kg > max(fs)])
    k.append(knew)
    t += 1
  # Output part of the algorithm
  return Xi[np.argmax(fx)]