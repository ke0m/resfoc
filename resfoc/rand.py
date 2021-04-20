"""
Utility functions to help with random generation

@author: Joseph Jennings
@version: 2020.02.07
"""
import numpy as np

def sign():
  """ Returns either a positive or negative sign """
  return np.random.choice([1,-1])

def randfloat(minf,maxf):
  """ 
  Returns a random float within a range

  Parameters:
    minf - minimum number in the range
    maxf - maximum number in the range
  """
  assert(maxf > minf),"maxf must be greater than minf"
  return np.random.rand()*(maxf-minf) + minf
