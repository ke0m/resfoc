"""
Forward and inverse cosine transform of N-dimensional arrays
@author: Joseph Jennings
@version: 2020.04.02
"""
import numpy as np
from resfoc.ficosft import fwdcosft,invcosft

def cosft(data,verb=False,**kwargs):
  """
  Forward cosine transform

  Parameters
    data  - Input numpy (up to 4D supported at the moment)
    axis0 - Flag indicating whether to compute transform along axis0 (0 or 1) [None]
    axis1 - Flag indicating whether to compute transform along axis1 (0 or 1) [None]
    axis2 - Flag indicating whether to compute transform along axis2 (0 or 1) [None]
    axis3 - Flag indicating whether to compute transform along axis3 (0 or 1) [None]
    verb  - Verbosity flag for verbose output [False]

  Returns the cosine transformed array
  """
  if(data.dtype != np.zeros(1,dtype='float32').dtype):
    raise Exception("Input data must be of type 'float32'")

  # Create the necessary inputs for the fwdcosft function
  dim   = len(data.shape)
  ns    = np.ones (4,dtype='int32')
  s     = np.zeros(4,dtype='int32')
  signs = np.zeros(4,dtype='int32')

  dim1 = -1
  for i in range(dim):
    ns[i] = data.shape[dim-i-1]
    if(1==ns[i] or kwargs.get("axis%d"%(dim-i-1),None) is None): 
      signs[i] = 0
    else:
      signs[i] = 1
    if(signs[i] == 1): 
      dim1 = i

  n1 = n2 = 1
  for i in range(dim):
    if(i <= dim1):
      s[i] = n1
      n1 *= ns[i]
    else:
      n2 *= ns[i]

  #TODO: it is probably better to not make a copy and write in place
  tmp = np.copy(data)
  fwdcosft(dim1,n1,n2,ns,signs,s,tmp,verb)

  return tmp

def icosft(data,verb=False,**kwargs):
  """
  Inverse cosine transform

  Parameters
    data  - Input numpy (up to 4D supported at the moment)
    axis0 - Flag indicating whether to compute transform along axis0 (0 or 1) [0]
    axis1 - Flag indicating whether to compute transform along axis1 (0 or 1) [0]
    axis2 - Flag indicating whether to compute transform along axis2 (0 or 1) [0]
    axis3 - Flag indicating whether to compute transform along axis3 (0 or 1) [0]
    verb  - Verbosity flag for verbose output [False]

  Returns the inverse cosine transformed array
  """
  if(data.dtype != np.zeros(1,dtype='float32').dtype):
    raise Exception("Input data must be of type 'float32'")

  # Create the necessary inputs for the invcosft function
  dim   = len(data.shape)
  ns    = np.ones (4,dtype='int32')
  s     = np.zeros(4,dtype='int32')
  signs = np.zeros(4,dtype='int32')

  dim1 = -1
  for i in range(dim):
    ns[i] = data.shape[dim-i-1]
    if(1==ns[i] or kwargs.get("axis%d"%(dim-i-1),None) is None): 
      signs[i] = 0 
    else:
      signs[i] = 1
    if(signs[i] == 1): 
      dim1 = i 

  n1 = n2 = 1 
  for i in range(dim):
    if(i <= dim1):
      s[i] = n1
      n1 *= ns[i]
    else:
      n2 *= ns[i]

  tmp = np.copy(data)
  invcosft(dim1,n1,n2,ns,signs,s,tmp,verb)

  return tmp 

def samplings(data,dsin):
  """ Computes the cosine transformed samplings """
  ns = data.shape
  ndim = len(ns)
  ds = []
  for idim in range(ndim):
    ds.append(1/(2*next_fast_size(ns[idim]-1)*dsin[idim]))

  return ds

def next_fast_size(n):
  """ Gets the next fastest size of the cosine transform """
  while(1):
    m = n 
    while( (m%2) == 0 ): m/=2
    while( (m%3) == 0 ): m/=3
    while( (m%5) == 0 ): m/=5
    if(m<=1):
      break
    n += 1

  return n

