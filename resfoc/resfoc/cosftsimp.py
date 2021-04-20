"""
Simple, inefficient cosine transform of up to 4D data
@author: Joseph Jennings
@version: 2020.04.02
"""
import numpy as np
from genutils.ptyprint import printprogress

## Main functions
def cosft(dat,axis1=None,axis2=None,axis3=None,axis4=None):
  """ Computes the forward cosine transform just as in Madagascar """
  if(len(dat.shape) == 1):
    return cosft1d(dat).astype('float32')
  elif(len(dat.shape) == 2):
    return cosft2d(dat,axis1,axis2).astype('float32')
  elif(len(dat.shape) == 3):
    return cosft3d(dat,axis1,axis2,axis3).astype('float32')
  elif(len(dat.shape) == 4):
    return cosft4d(dat,axis1,axis2,axis3,axis4).astype('float32')
  else:
    print("Cosine transform > 4D not yet implemented")
    return

def icosft(dat,axis1=None,axis2=None,axis3=None,axis4=None,verb=False):
  """ Computes the inverse cosine transform """
  if(len(dat.shape) == 1):
    return icosft1d(dat).astype('float32')
  elif(len(dat.shape) == 2):
    return icosft2d(dat,axis1,axis2).astype('float32')
  elif(len(dat.shape) == 3):
    return icosft3d(dat,axis1,axis2,axis3).astype('float32')
  elif(len(dat.shape) == 4):
    return icosft4d(dat,axis1,axis2,axis3,axis4,verb=False).astype('float32')
  else:
    print("Inverse cosine transform > 4D not yet implemented")
    return

## 1D Transforms
def cosft1d(sig):
  """ Computes the forward 1D cosine transform """
  # Get sizes
  n1 = sig.shape[0]
  nt = 2*next_fast_size(n1-1)
  nw = int(nt/2) + 1
  # Prepare input
  p = np.zeros(nt)
  p[0:n1] = sig[:]
  p[n1:nw] = 0.0
  p[nw:nt] = p[nt-nw:0:-1]
  # Compute cosine transform
  return np.real(np.fft.rfft(p))[0:n1]

def icosft1d(sig):
  """ Computes the inverse 1D cosine transform """
  # Get sizes
  n1 = sig.shape[0]
  nt = 2*next_fast_size(n1-1)
  nw = int(nt/2) + 1
  # Prepare input
  p = np.zeros(nw)
  p[0:n1] = sig[:]
  p[n1:nw] = 0.0
  # Compute inverse cosine transform
  return np.real(np.fft.irfft(p))[0:n1]

## 2D transforms
def cosft2d(img,axis1=None,axis2=None):
  """ 2D cosine transform along axis 1 and/or 2 """
  if(axis1==None and axis2==None):
    return img
  if(axis1 and not axis2):
    return cosft2d1(img)
  elif(axis2 and not axis1):
    return cosft2d2(img)
  else:
    ft  =  cosft2d1(img)
    return cosft2d2(ft)

def icosft2d(img,axis1=None,axis2=None):
  """ Inverse 2D cosine transform along axis 1 and/or 2 """
  if(axis1==None and axis2==None):
    return img
  if(axis1 and not axis2):
    return icosft2d1(img)
  elif(axis2 and not axis1):
    return icosft2d2(img)
  else:
    ift  = icosft2d1(img)
    return icosft2d2(ift)

def cosft2d1(img):
  """ 2D cosine transform along axis 1 (cols/slow) """
  # Get sizes
  n1,n2 = img.shape
  nt = 2*next_fast_size(n1-1)
  nw = int(nt/2) + 1
  # Prepare the input
  pimg = np.zeros([nt,n2])
  pimg[0:n1,:] = img[:,:]
  pimg[n1:nw,:] = 0.0
  pimg[nw:nt,:] = pimg[nt-nw:0:-1]
  # Compute the cosine transform
  return np.real(np.fft.rfft(pimg,axis=0))[0:n1,:]

def icosft2d1(img):
  """ 2D inverse cosine transform along axis 1 (cols/slow) """
  # Get sizes
  n1,n2 = img.shape
  nt = 2*next_fast_size(n1-1)
  nw = int(nt/2) + 1
  # Prepare the input
  pimg = np.zeros([nw,n2])
  pimg[0:n1,:] = img[:,:]
  pimg[n1:nw,:] = 0.0
  # Compute the inverse cosine transform
  return np.real(np.fft.irfft(pimg,axis=0))[0:n1,:]

def cosft2d2(img):
  """ 2D cosine transform along axis 2 (rows/fast) """
  # Get sizes
  n1,n2 = img.shape
  nt = 2*next_fast_size(n2-1)
  nw = int(nt/2) + 1
  # Prepare the input
  pimg = np.zeros([n1,nt])
  pimg[:,0:n2] = img[:,:]
  pimg[:,n2:nw] = 0.0
  pimg[:,nw:nt] = pimg[:,nt-nw:0:-1]
  # Compute the cosine transform
  return np.real(np.fft.rfft(pimg,axis=1))[:,0:n2]

def icosft2d2(img):
  """ 2D inverse cosine transform along axis 2 (rows/fast) """
  # Get sizes
  n1,n2 = img.shape
  nt = 2*next_fast_size(n2-1)
  nw = int(nt/2) + 1
  # Prepare the input
  pimg = np.zeros([n1,nw])
  pimg[:,0:n2] = img[:,:]
  pimg[:,n2:nw] = 0.0
  # Compute the inverse cosine transform
  return np.real(np.fft.irfft(pimg,axis=1))[:,0:n2]

## 3D transforms
def cosft3d(cub,axis1=None,axis2=None,axis3=None):
  """ 3D cosine transform """
  if(axis1==None and axis2==None and axis3==None):
    return cub
  # Single axes transforms
  if(axis1 and not axis2 and not axis3):
    return cosft3d1(cub)
  elif(axis2 and not axis1 and not axis3):
    return cosft3d2(cub)
  elif(axis3 and not axis1 and not axis2):
    return cosft3d3(cub)
  # Multi-axes transforms
  elif(axis1 and axis2 and not axis3):
    ft  =  cosft3d1(cub)
    return cosft3d2(ft)
  elif(axis1 and axis3 and not axis2):
    ft  =  cosft3d1(cub)
    return cosft3d3(ft)
  elif(axis2 and axis3 and not axis1):
    ft  =  cosft3d2(cub)
    return cosft3d3(ft)
  else:
    ft1  = cosft3d1(cub)
    ft12 = cosft3d2(ft1)
    return cosft3d3(ft12)

def icosft3d(cub,axis1=None,axis2=None,axis3=None):
  """ 3D inverse cosine transform """
  if(axis1==None and axis2==None and axis3==None):
    return cub
  # Single axes inverse transforms
  if(axis1 and not axis2 and not axis3):
    return icosft3d1(cub)
  elif(axis2 and not axis1 and not axis3):
    return icosft3d2(cub)
  elif(axis3 and not axis1 and not axis2):
    return icosft3d3(cub)
  # Multi-axes inverse transforms
  elif(axis1 and axis2 and not axis3):
    ift  = icosft3d1(cub)
    return icosft3d2(ift)
  elif(axis1 and axis3 and not axis2):
    ift  = icosft3d1(cub)
    return icosft3d3(ift)
  elif(axis2 and axis3 and not axis1):
    ift  = icosft3d2(cub)
    return icosft3d3(ift)
  else:
    ift1  = icosft3d1(cub)
    ift12 = icosft3d2(ift1)
    return  icosft3d3(ift12)

def cosft3d1(cub):
  """ Cosine transform along axis 1 (slowest) """
  # Get sizes
  n1,n2,n3 = cub.shape
  nt = 2*next_fast_size(n1-1)
  nw = int(nt/2) + 1
  # Prepare input
  pcub = np.zeros([nt,n2,n3])
  pcub[0:n1,:,:] = cub[:,:,:]
  pcub[n1:nw,:,:]  = 0.0
  pcub[nw:nt,:,:]  = pcub[nt-nw:0:-1]
  # Compute cosine transform
  return np.real(np.fft.rfft(pcub,axis=0))[0:n1,:,:]

def icosft3d1(cub):
  """ Inverse cosine transform along axis 1 (slowest) """
  # Get sizes
  n1,n2,n3 = cub.shape
  nt = 2*next_fast_size(n1-1)
  nw = int(nt/2) + 1
  # Prepare input
  pcub = np.zeros([nw,n2,n3])
  pcub[0:n1,:,:]  = cub[:,:,:]
  pcub[n1:nw,:,:] = 0.0
  # Compute inverse cosine transform
  return np.real(np.fft.irfft(pcub,axis=0))[0:n1,:,:]

def cosft3d2(cub):
  """ Cosine transform along axis 2 """
  # Get sizes
  n1,n2,n3 = cub.shape
  nt = 2*next_fast_size(n2-1)
  nw = int(nt/2) + 1
  # Prepare input
  pcub = np.zeros([n1,nt,n3])
  pcub[:,0:n2,:] = cub[:,:,:]
  pcub[:,n2:nw,:] = 0.0
  pcub[:,nw:nt,:] = pcub[:,nt-nw:0:-1,:]
  # Compute cosine transform
  return np.real(np.fft.rfft(pcub,axis=1))[:,0:n2,:]

def icosft3d2(cub):
  """ Inverse cosine transform along axis 2 """
  # Get sizes
  n1,n2,n3 = cub.shape
  nt = 2*next_fast_size(n2-1)
  nw = int(nt/2) + 1
  # Prepare input
  pcub = np.zeros([n1,nw,n3])
  pcub[:,0:n2,:] = cub[:,:,:]
  pcub[:,n2:nw,:] = 0.0
  # Compute inverse cosine transform
  return np.real(np.fft.irfft(pcub,axis=1))[:,0:n2,:]

def cosft3d3(cub):
  """ Cosine transform along axis 3 (fastest) """
  # Get sizes
  n1,n2,n3 = cub.shape
  nt = 2*next_fast_size(n3-1)
  nw = int(nt/2) + 1
  # Prepare input
  pcub = np.zeros([n1,n2,nt])
  pcub[:,:,0:n3] = cub[:,:,:]
  pcub[:,:,n3:nw] = 0.0
  pcub[:,:,nw:nt] = pcub[:,:,nt-nw:0:-1]
  # Compute cosine transform
  return np.real(np.fft.rfft(pcub,axis=2))[:,:,0:n3]

def icosft3d3(cub):
  """ Inverse cosine transform along axis 3 """
  # Get sizes
  n1,n2,n3 = cub.shape
  nt = 2*next_fast_size(n2-1)
  nw = int(nt/2) + 1
  # Prepare input
  pcub = np.zeros([n1,n2,nw])
  pcub[:,:,0:n3] = cub[:,:,:]
  pcub[:,:,n3:nw] = 0.0
  # Compute inverse cosine transform
  return np.real(np.fft.irfft(pcub,axis=2))[:,:,0:n3]

def cosft4d(hcub,axis1=None,axis2=None,axis3=None,axis4=None):
  """ 4D cosine transform """
  if(axis1==None and axis2==None and axis3==None and axis4==None):
    return hcub
  # Single axes transforms
  if(axis1 and not axis2 and not axis3 and not axis4):
    return cosft4d1(hcub)
  elif(axis2 and not axis1 and not axis3 and not axis4):
    return cosft4d2(hcub)
  elif(axis3 and not axis1 and not axis2 and not axis4):
    return cosft4d3(hcub)
  elif(axis4 and not axis1 and not axis2 and not axis3):
    return cosft4d2(hcub)
  # Multi-axes transforms
  elif(axis1 and axis2 and axis3 and not axis4):
    ft1 =  cosft4d1(hcub)
    ft12 = cosft4d2(ft1)
    return cosft4d3(ft12)
  else:
    print("This type of 4D cosft has not been yet implemented")
    return

def icosft4d(hcub,axis1=None,axis2=None,axis3=None,axis4=None,verb=False):
  """ 4D inverse cosine transform """
  # Single axes inverse transforms
  if(axis1 and not axis2 and not axis3 and not axis4):
    return icosft4d1(hcub)
  elif(axis2 and not axis1 and not axis3 and not axis4):
    return icosft4d2(hcub)
  elif(axis3 and not axis1 and not axis2 and not axis4):
    return icosft4d3(hcub)
  elif(axis4 and not axis1 and not axis2 and not axis3):
    return icosft4d2(hcub)
  # Multi-axes transforms
  elif(axis1 and axis2 and axis3 and not axis4):
    ift1  = icosft4d1(hcub)
    ift12 = icosft4d2(ift1)
    return  icosft4d3(ift12)
  elif(axis2 and axis3 and axis4 and not axis1):
    printprogress("axes:", 0, 3)
    ift2   = icosft4d2(hcub)
    printprogress("axes:", 1, 3)
    ift23  = icosft4d3(ift2)
    printprogress("axes:", 2, 3)
    ift234 = icosft4d4(ift23)
    printprogress("axes:", 3, 3)
    return  ift234
  else:
    print("This type of 4D icosft has not yet been implemented")
    return

def cosft4d1(hcub):
  """ Cosine transform along axis 1 (slowest) """
  # Get sizes
  n1,n2,n3,n4 = hcub.shape
  nt = 2*next_fast_size(n1-1)
  nw = int(nt/2) + 1
  # Prepare input
  phcub = np.zeros([nt,n2,n3,n4])
  phcub[0:n1,:,:,:] = hcub[:,:,:,:]
  phcub[n1:nw,:,:,:]  = 0.0
  phcub[nw:nt,:,:,:]  = phcub[nt-nw:0:-1]
  # Compute cosine transform
  return np.real(np.fft.rfft(phcub,axis=0))[0:n1,:,:,:]

def icosft4d1(hcub):
  """ Inverse cosine transform along axis 1 (slowest) """
  # Get sizes
  n1,n2,n3,n4 = hcub.shape
  nt = 2*next_fast_size(n1-1)
  nw = int(nt/2) + 1
  # Prepare input
  phcub = np.zeros([nw,n2,n3,n4])
  phcub[0:n1,:,:,:]  = hcub[:,:,:,:]
  phcub[n1:nw,:,:,:] = 0.0
  # Compute inverse cosine transform
  return np.real(np.fft.irfft(phcub,axis=0))[0:n1,:,:,:]

def cosft4d2(hcub):
  """ Cosine transform along axis 2 """
  # Get sizes
  n1,n2,n3,n4 = hcub.shape
  nt = 2*next_fast_size(n2-1)
  nw = int(nt/2) + 1
  # Prepare input
  phcub = np.zeros([n1,nt,n3,n4])
  phcub[:,0:n2,:,:] = hcub[:,:,:,:]
  phcub[:,n2:nw,:,:]  = 0.0
  phcub[:,nw:nt,:,:]  = phcub[:,nt-nw:0:-1,:,:]
  # Compute cosine transform
  return np.real(np.fft.rfft(phcub,axis=1))[:,0:n2,:,:]

def icosft4d2(hcub):
  """ Inverse cosine transform along axis 2 """
  # Get sizes
  n1,n2,n3,n4 = hcub.shape
  nt = 2*next_fast_size(n2-1)
  nw = int(nt/2) + 1
  # Prepare input
  phcub = np.zeros([n1,nw,n3,n4])
  phcub[:,0:n2,:,:]  = hcub[:,:,:,:]
  phcub[:,n2:nw,:,:] = 0.0
  # Compute inverse cosine transform
  return np.real(np.fft.irfft(phcub,axis=1))[:,0:n2,:,:]

def cosft4d3(hcub):
  """ Cosine transform along axis 3 """
  # Get sizes
  n1,n2,n3,n4 = hcub.shape
  nt = 2*next_fast_size(n3-1)
  nw = int(nt/2) + 1
  # Prepare input
  phcub = np.zeros([n1,n2,nt,n4])
  phcub[:,:,0:n3,:] = hcub[:,:,:,:]
  phcub[:,:,n3:nw,:]  = 0.0
  phcub[:,:,nw:nt,:]  = phcub[:,:,nt-nw:0:-1,:]
  # Compute cosine transform
  return np.real(np.fft.rfft(phcub,axis=2))[:,:,0:n3,:]

def icosft4d3(hcub):
  """ Inverse cosine transform along axis 3 """
  # Get sizes
  n1,n2,n3,n4 = hcub.shape
  nt = 2*next_fast_size(n3-1)
  nw = int(nt/2) + 1
  # Prepare input
  phcub = np.zeros([n1,n2,nw,n4])
  phcub[:,:,0:n3,:]  = hcub[:,:,:,:]
  phcub[:,:,n3:nw,:] = 0.0
  # Compute inverse cosine transform
  return np.real(np.fft.irfft(phcub,axis=2))[:,:,0:n3,:]

def cosft4d4(hcub):
  """ Cosine transform along axis 4 (fastest) """
  # Get sizes
  n1,n2,n3,n4 = hcub.shape
  nt = 2*next_fast_size(n4-1)
  nw = int(nt/2) + 1
  # Prepare input
  phcub = np.zeros([n1,n2,n3,nt])
  phcub[:,:,:,0:n4] = hcub[:,:,:,:]
  phcub[:,:,:,n4:nw]  = 0.0
  phcub[:,:,:,nw:nt]  = phcub[:,:,:,nt-nw:0:-1]
  # Compute cosine transform
  return np.real(np.fft.rfft(phcub,axis=3))[:,:,:,0:n4]

def icosft4d4(hcub):
  """ Inverse cosine transform along axis 4 (fastest) """
  # Get sizes
  n1,n2,n3,n4 = hcub.shape
  nt = 2*next_fast_size(n4-1)
  nw = int(nt/2) + 1
  # Prepare input
  phcub = np.zeros([n1,n2,n3,nw])
  phcub[:,:,:,0:n4]  = hcub[:,:,:,:]
  phcub[:,:,:,n4:nw] = 0.0
  # Compute inverse cosine transform
  return np.real(np.fft.irfft(phcub,axis=3))[:,:,:,0:n4]

def samplings(dat,dsin):
  """ Computes the cosine transformed samplings """
  ns = dat.shape
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

