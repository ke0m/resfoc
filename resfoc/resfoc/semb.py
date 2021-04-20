"""
Functions for computing and picking semblance

@author: Joseph Jennings
@version: 2020.06.20
"""
import numpy as np
from resfoc.gain import agc
from joblib import Parallel,delayed
from scaas.trismooth import smooth,smoothop
from resfoc.pickscan import pickscan
from opt.linopt.essops.weight import weight
from opt.linopt.cd import cd

def rho_semb(stormang,gagc=True,norm=True,rectz=10,rectro=3,nthreads=1):
  """
  Computes semblance from residually migrated angle gathers

  Parameters:
    stormang - Stolt resiudally migrated angle gathers [nro,nx,na,nz]
    gagc     - Apply agc [True]
    rectz    - Smoothing window along z direction [10 points]
    rectro   - Smoothing window along rho direction [3 points]

  Returns a semblance cube [nx,nro,nz]
  """
  # Get dimensions
  nro,nx,na,nz = stormang.shape
  # Compute agc
  if(gagc):
    angs = np.asarray(Parallel(n_jobs=nthreads)(delayed(agc)(stormang[iro]) for iro in range(nro)))
  else:
    angs = stormang

  # Compute semblance
  stackg  = np.sum(angs,axis=2)
  stacksq = stackg*stackg
  num = smooth(stacksq.astype('float32'),rect1=rectz,rect3=rectro)

  sqstack = np.sum(angs*angs,axis=2)
  den = smooth(sqstack.astype('float32'),rect1=rectz,rect3=rectro)

  semb = num/den

  sembt = np.transpose(semb,(1,0,2)) # [nro,nx,nz] -> [nx,nro,nz]

  if(norm):
    sembt /= np.max(sembt)

  return sembt

def pick(semb,opar,dpar,vel0=None,norm=False,rectz=40,rectx=20,an=1.0,gate=3,niter=100,verb=False):
  """
  Computes semblance picks for an input semblance panel

  Parameters:
    semb  - input semblance panels [nx,npar,nz]
    opar  - origin of scan parameter
    dpar  - sampling of scan parameter
    vel0  - initial pick at surface (surface velocity) [opar]
    norm  - normalize input panels [True]
    rectz - length of smoothing window in z [40 points]
    rectx - length of smoothing window in x [20 points]
    an    - axes anisotropy [1.]
    gate  - picking gate [3]
    niter - number of iterations for smooth division [100]
    verb  - inversion verbosity flag [False]

  An image of the semblance picks [nx,nz]
  """
  # Get dimensions
  nx,npar,nz = semb.shape
  # Allocate arrays
  pck2 = np.zeros([nx,nz],dtype='float32')
  ampl = np.zeros([nx,nz],dtype='float32')
  pcko = np.zeros([nx,nz],dtype='float32')

  if(vel0 is None):
    vel0 = opar

  # Find semblance picks
  pickscan(an,gate,norm,vel0,opar,dpar,nz,npar,nx,semb,pck2,ampl,pcko)

  # Return pck2 if no smoothing
  if(rectz == 1 and rectx == 1):
    opicks = opar + pck2*dpar
    return opicks

  # Build shaping operator
  smop = smoothop([nx,nz],rect1=rectz,rect2=rectx)

  # Weight (element-wise multiplication) operator
  wop = weight(ampl)

  # Initial model
  sm0 = np.zeros(ampl.shape,dtype='float32')

  # Smooth the picks via smooth division
  smf = cd(wop,pcko,sm0,shpop=smop,niter=niter,verb=verb)

  opicks = smf + vel0

  return opicks

