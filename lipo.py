# Author Ioannis Athanasiadis (ath.ioannis94@gmail.com) 2018
import numpy as np

__all__ = ["lipo", "adalipo"]

def lipo(n, k, X, f, MAX_ITERATIONS_PER_T = 100):
  """
    This is the lipo algorithm as described in section 3
    of the work of Malherbe and Vayatis.
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

def _len(X):
  return np.prod(np.shape(X))

def adalipo(n, K, X, f, p):
  """
    This the adalipo algorithm described in section 4
    of the work of Malherbe and Vayatis. It does not require a specific k
    value as lipo, but a grid of values where to search for it.
    Args:
      n : number of iterations
      K : grid of lipschitz constants (list or numpy array)
      X : possible inputs of f
      f : the lipschitz function which to find maxima (as a function object)
      p : propability of exploitation vs (1-p) propability of exploration
    Returns:
        The estimated value which maximizes f.
  """
  # initialize I as a numpy array of input X
  # for faster computations
  # if(len(np.shape(X)) > 1):
  #   I = np.squeeze(X)
  # else:
  I = np.array(X)
  if(len(I.shape) == 1):
    I = np.expand_dims(I, -1)
  # make sure that K will be a numpy array
  Kg = np.sort(np.array(K))
  # Initialization part of the algorithm
  x1 = np.squeeze(I[np.random.randint(I.shape[0], size=1), :], axis=0)
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
      Xnew = np.squeeze(I[np.random.randint(I.shape[0], size=1), :])
    else: # exploitation
      mx = np.max(fx, axis=-1)
      # if(len(np.shape(I)) > 1):
        # Ii = np.squeeze(Xi)
      # else:
      Ii = np.array(Xi)
      if(len(Ii.shape) == 1):
        Ii = np.expand_dims(Ii, 0)
      # print(Ii)
      # l contains the norm-2 distance between the points the algorithm
      # has explored and the points in the possible inputs of f
      l = np.zeros((len(Ii), len(I)))
      for idx_i, Iii in enumerate(Ii):
        s_outer = np.subtract.outer(np.expand_dims(Iii, 0), I)
        d = [s_outer[0, i, :, i] for i in xrange(np.shape(I)[-1])]
        l[idx_i,:] = np.linalg.norm(d, axis=0)
      # w is an estimation of the term inside the min inside 
      # the Lemma 8 after definition 7
      w = knew * l
      nfx = fx[:t]
      for i in xrange(np.shape(w)[1]):
        w[:,i] += nfx
      Xkt = X[np.min(w, axis=0) >= mx, :]
      if(_len(Xkt) != 0): # exploit
        Xnew = np.squeeze(Xkt[np.random.randint(Xkt.shape[0], size=1), :])
      else: # explore again...
        Xnew = np.squeeze(I[np.random.randint(I.shape[0], size=1), :])
    if(len(np.shape(Xnew)) < 1):
      Xnew = np.expand_dims(Xnew, -1)
    Xi.append(Xnew)
    fx[t] = f(Xnew)
    df = np.subtract.outer(fx[:t+1], fx[:t+1])
    fs = []
    idx = 0
    # fs = np.zeros(((len(Xi)) * (len(Xi) - 1), ))
    ndf = np.linalg.norm(np.expand_dims(df, -1), axis=-1)
    # print(np.shape(Xi), Xi)
    dX = np.subtract.outer(Xi, Xi)
    # print(np.shape(dX), np.shape(Xi))
    dx = np.array([dX[:,i,:,i] for i in xrange(np.shape(dX)[-1])])
    ndx = np.linalg.norm(dx, axis=0)
    for i in xrange(np.shape(ndf)[0]):
      fs.extend(ndf[i,:i] / ndx[i,:i])
      if(i+1 != np.shape(ndf)):
        fs.extend(ndf[i,i+1:] / ndx[i,i+1:])
    try:
      knew = np.min(Kg[Kg > np.max(fs)])
    except:
      print("function has greater lipschitz constants than those provided")
      return ValueError
    k.append(knew)
    t += 1
  # Output part of the algorithm
  return np.squeeze(Xi[np.argmax(fx)])