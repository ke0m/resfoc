"""
Gain functions for better display of data and images

@author: Joseph Jennings
@version: 2020.03.30
"""
import numpy as np
from scaas.trismooth import smooth

def tpow(dat,dt,tpow,ot=0.0,norm=True,transp=False):
  """
  Applies a t^pow gain

  Parameters
    dat  - input array of order [nro,nh,nt,nx] where nro and nh are optional
    dt   - the sampling rate of the data
    tpow - the power of time
    ot   - origin of the time axis [0.0]
    norm - normalize the gain function before applying
    transp - whether to transpose the input array from [nro,nh,nx,nt] -> [nro,nh,nt,nx]

  Returns the gained array of same dimensions as input
  """
  #TODO: 4D not yet implemented
  if(len(dat.shape) == 4):
    # [nro,nh,nx,nt] -> [nro,nh,nt,nx]
    if(transp):
      dat = np.transpose(dat,(0,1,3,2))
      tp = tpow4d(dat,dt,tpow,ot,norm)
      return np.ascontiguousarray(np.transpose(tp,(0,1,3,2)))
    else:
      return tpow4d(dat,dt,tpow,ot,norm)
  elif(len(dat.shape) == 3):
    # [nh,nx,nt] -> [nh,nt,nx]
    if(transp):
      dat = np.transpose(dat,(0,2,1))
      tp = tpow3d(dat,dt,tpow,ot,norm)
      return np.ascontiguousarray(np.transpose(tp,(0,2,1)))
    else:
      return tpow3d(dat,dt,tpow,ot,norm)
  else:
    if(transp):
      return np.ascontiguousarray(tpow2d(dat.T,dt,tpow,ot,norm).T)
    else:
      return tpow2d(dat.T,dt,tpow,ot,norm)

def tpow2d(dat,dt,tpow,ot=0.0,norm=True):
  """ Applies a t^pow gain to the input 2D array dat """
  # Get the shape of the data
  nx = dat.shape[1]; nt = dat.shape[0]
  # Build the t function
  if(ot != 0.0):
    t = np.linspace(ot,ot + (nt-1)*dt, nt)
  else:
    ot = dt
    t = np.linspace(ot,ot + (nt-1)*dt, nt)
  tp = np.power(t,tpow)
  # Normalize by default
  if(norm): tp = tp/np.max(tp)
  # Replicate it across the other axes
  tpx   = np.tile(np.array([tp]).T,(1,nx))
  # Scale the data
  return(dat*tpx).astype('float32')

def tpow3d(dat,dt,tpow,ot=0.0,norm=True):
  """ Applies a t^pow gain to the input array dat """
  # Get the shape of the data
  nx = dat.shape[2]; nt = dat.shape[1]; nh = dat.shape[0]
  # Build the t function
  if(ot != 0.0):
    t = np.linspace(ot,ot + (nt-1)*dt, nt)
  else:
    ot = dt
    t = np.linspace(ot,ot + (nt-1)*dt, nt)
  tp = np.power(t,tpow)
  # Normalize by default
  if(norm): tp = tp/np.max(tp)
  # Replicate it across the other axes
  tpx   = np.tile(np.array([tp]).T,(1,nx))
  tpxr  = np.tile(tpx,(nh,1,1))
  # Scale the data
  return (dat*tpxr).astype('float32')

def agc(dat,rect1=125,transp=False):
  """
  Applies an automatic gain control (AGC) to the data/image
  Applies it trace by trace (assumes t or z is the fast axis)

  Parameters:
    dat    - the input data/image [nx,nt/nz]
    rect1  - size of AGC window along the fast axis
    transp - transpose a 2D image so that t/z is the fast axis

  Returns the gained data
  """
  if(transp):
    dat = np.ascontiguousarray(dat.T)
  # First compute the absolute value of the data
  databs = np.abs(dat)
  # Smooth the absolute value
  databssm = smooth(databs,rect1=rect1)
  # Divide by the smoothed amplitude
  idx = databssm <= 0
  databssm[idx] = 1
  return dat/databssm

