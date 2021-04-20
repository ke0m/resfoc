"""
Metrics for quantitative evaluation of
deep learning predictions
@author: Joseph Jennings
@version: 2020.04.14
"""

def iou():
  pass

def intersect(a,b):
  """
  Computes the intersection between two thresholded numpy arrays

  Parameters
    a - input numpy array
    b - input numpy array
 
  Returns a numpy array containing the intersection between the two arrays
  """
  n = len(a)
  if(n != len(b)):
    raise Exception("Input arrays must be same length")
  aidx = a == 1.0; bidx = b == 1.0
  return np.logical_and(aidx,bidx)

def union(a,b):
  pass

