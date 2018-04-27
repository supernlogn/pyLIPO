# Author Ioannis Athanasiadis (ath.ioannis94@gmail.com) 2018
import numpy as np


__all__ = ["plipo"]

def plipo(n, k, X, f, p, MAX_ITERATIONS_PER_T = 100):
  """
    This is the plipo algorithm which is a parallel version f the lipo as
    described in section of the work of Malherbe and Vayatis.
    Args:
      n : number of parallel iterations
      K : The specific lipschitz constant, best to use the smallest possible
      X : possible inputs of f
      f : the lipschitz function which to find maxima (as a function object)
      p : number of parallel processes to launch per iteration
      MAX_ITERATIONS_PER_T : (optional) max iterations to do per timestep t, the algorithm can find itself
                              in an endless loop cause it depends to propability in order to go to the next timestep.
    Returns:
      The estimated value which maximizes f.
  """
  def p_func(els_to_calculate):
    p_func_vals = []
    for el_in in list(els_to_calculate):
      p_func_vals.append(f(el_in))
    return p_func_vals
  # initialization part of the algorithm
  x1 = np.random.choice(X)
  fx = [] # holds the functions values computed so far...
  fx.append(f(x1))
  Xi = [x1]  # holds the functions input values explored so far...
  t = 1
  it_per_t = 0
  # parallel Iterations part of the algorithm
  while(t < n and it_per_t < MAX_ITERATIONS_PER_T):
    Xnews = np.random.choice(
                np.delete(X, np.where(np.isin(X, Xi))),
                (p))
    nfx = np.array(fx)
    it_per_t += len(Xnews)
    # TODO: the line below required a dimension more for its valid execution
    idx = np.min(nfx + k * np.linalg.norm(Xnews - Xi)) > np.max(fx)
    els_to_calculate = np.where(
                    idx,
                    Xnew,
                    np.nan)
    fx.extend(p_func(els_to_calculate))
    Xi.extend(Xnew)
  return Xi[np.argmax(np.array(fx))]
