"""
Functions that create "standard" velocity models
Current models are:
  1. Layered v(z) with three faults
  2. Heavily faulted models with complex velocity

@author: Joseph Jennings
@version: 2020.05.01
"""
import numpy as np
import velocity.mdlbuild as mdlbuild
from scaas.wavelet import ricker
from genutils.ptyprint import progressbar, create_inttag
import genutils.rand as rndut
import deeplearn.utils as dlut
from scipy.ndimage import gaussian_filter
from genutils.signal import bandpass

def velfaultsrandom(nz=512,nx=1024,ny=20,dz=12.5,dx=25.0,nlayer=20,
                    minvel=1600,maxvel=5000,rect=0.5,
                    verb=True,**kwargs):
  """
  Builds a 2D highly faulted and folded velocity model.
  Returns the velocity model, reflectivity, fault labels and a zero-offset image

  Parameters:
    nz     - number of depth samples [512]
    nx     - number of lateral samples [1024]
    dz     - depth sampling interval [25.0]
    dx     - lateral sampling interval [25.0]
    nlayer - number of deposited layers (there exist many fine layers within a deposit) [20]
    minvel - minimum velocity in model [1600]
    maxvel - maximum velocity in model [5000]
    rect   - length of gaussian smoothing [0.5]
    verb   - verbosity flag [True]

  Returns
    The velocity, reflectivity, fault label and image all of size [nx,nz]
  """
  # Internal model size
  nzi = 1000; nxi = 1000
  # Model building object
  mb = mdlbuild.mdlbuild(nxi,dx,ny,dy=dx,dz=dz,basevel=5000)

  # First build the v(z) model
  props = mb.vofz(nlayer,minvel,maxvel,npts=kwargs.get('nptsvz',2))

  # Specify the thicknesses
  thicks = np.random.randint(40,61,nlayer)

  # Determine when to fold the deposits
  sqlyrs = sorted(mb.findsqlyrs(3,nlayer,5))
  csq = 0

  dlyr = 0.05
  for ilyr in progressbar(range(nlayer), "ndeposit:", 40, verb=verb):
    mb.deposit(velval=props[ilyr],thick=thicks[ilyr],dev_pos=0.0,
               layer=kwargs.get('layer',150),layer_rand=0.00,dev_layer=dlyr)
    # Random folding
    if(ilyr in sqlyrs):
      if(sqlyrs[csq] < 15):
        # Random amplitude variation in the folding
        amp = np.random.rand()*(3000-500) + 500
        mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin',order=3)
      elif(sqlyrs[csq] >= 15 and sqlyrs[csq] < 18):
        amp = np.random.rand()*(1800-500) + 500
        mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin',order=3)
      else:
        amp = np.random.rand()*(500-300) + 300
        mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin')
      csq += 1

  # Water deposit
  mb.deposit(1480,thick=50,layer=150,dev_layer=0.0)

  # Smooth any unconformities
  mb.smooth_model(rect1=1,rect2=5,rect3=1)

  # Trim model before faulting
  mb.trim(0,1100)

  # Fault it up!
  azims = [0.0,180.0]
  fprs  = [True,False]

  # Large faults
  nlf = np.random.randint(2,5)
  for ifl in progressbar(range(nlf), "nlfaults:", 40, verb=verb):
    azim = np.random.choice(azims)
    fpr  = np.random.choice(fprs)
    xpos = rndut.randfloat(0.1,0.9)
    mb.largefault(azim=azim,begz=0.65,begx=xpos,begy=0.5,dist_die=4.0,tscale=6.0,fpr=fpr,twod=True)

  # Medium faults
  nmf = np.random.randint(3,6)
  for ifl in progressbar(range(nmf), "nmfaults:", 40, verb=verb):
    azim = np.random.choice(azims)
    fpr  = np.random.choice(fprs)
    xpos = rndut.randfloat(0.05,0.95)
    mb.mediumfault(azim=azim,begz=0.65,begx=xpos,begy=0.5,dist_die=4.0,tscale=3.0,fpr=fpr,twod=True)

  # Small faults (sliding or small)
  nsf = np.random.randint(5,10)
  for ifl in progressbar(range(nsf), "nsfaults:", 40, verb=verb):
    azim = np.random.choice(azims)
    fpr  = np.random.choice(fprs)
    xpos = rndut.randfloat(0.05,0.95)
    zpos = rndut.randfloat(0.2,0.5)
    mb.smallfault(azim=azim,begz=zpos,begx=xpos,begy=0.5,dist_die=4.0,tscale=2.0,fpr=fpr,twod=True)

  # Tiny faults
  ntf = np.random.randint(5,10)
  for ifl in progressbar(range(ntf), "ntfaults:", 40, verb=verb):
    azim = np.random.choice(azims)
    xpos = rndut.randfloat(0.05,0.95)
    zpos = rndut.randfloat(0.15,0.3)
    mb.tinyfault(azim=azim,begz=zpos,begx=xpos,begy=0.5,dist_die=4.0,tscale=2.0,twod=True)

  # Parameters for ricker wavelet
  nt = kwargs.get('nt',250); ot = 0.0; dt = kwargs.get('dt',0.001); ns = int(nt/2)
  amp = 1.0; dly = kwargs.get('dly',0.125)
  minf = kwargs.get('minf',60.0); maxf = kwargs.get('maxf',100.0)
  f = kwargs.get('f',None)

  # Get model
  vel = gaussian_filter(mb.vel[:,:nzi],sigma=rect).astype('float32')
  lbl = mb.get_label2d()[:,:nzi]

  # Resample to output size
  velr = dlut.resample(vel,[nx,nz],kind='quintic')
  lblr = dlut.thresh(dlut.resample(lbl,[nx,nz],kind='linear'),0)
  refr = mb.calcrefl2d(velr)

  # Create normalized image
  if(f is None):
    f = rndut.randfloat(minf,maxf)
  wav = ricker(nt,dt,f,amp,dly)
  img = dlut.normalize(np.array([np.convolve(refr[ix,:],wav) for ix in range(nx)])[:,ns:nz+ns])
  # Create noise
  nze = dlut.normalize(bandpass(np.random.rand(nx,nz)*2-1, 2.0, 0.01, 2, pxd=43))/rndut.randfloat(3,5)
  img += nze

  if(kwargs.get('transp',False) == True):
    velt = np.ascontiguousarray(velr.T).astype('float32')
    reft = np.ascontiguousarray(refr.T).astype('float32')
    imgt = np.ascontiguousarray(img.T).astype('float32')
    lblt = np.ascontiguousarray(lblr.T).astype('float32')
  else:
    velt = np.ascontiguousarray(velr).astype('float32')
    reft = np.ascontiguousarray(refr).astype('float32')
    imgt = np.ascontiguousarray(img).astype('float32')
    lblt = np.ascontiguousarray(lblr).astype('float32')

  if(kwargs.get('km',True)): velt /= 1000.0

  return velt,reft,imgt,lblt

def layeredfaults2d(nz=512,nx=1000,dz=12.5,dx=25.0,nlayer=21,minvel=1600,maxvel=3000,rect=0.5,
                    nfx=3,ofx=0.4,dfx=0.1,ofz=0.3):
  """
  Builds a 2D layered, v(z) fault model.
  Returns the velocity model, reflectivity, fault labels
  and a zero-offset image

  Parameters:
    nz     - number of depth samples [512]
    nx     - number of lateral samples[1000]
    dz     - depth sampling interval [12.5]
    dx     - lateral sampling interval [12.5]
    nlayer - number of deposited layers (there exist many fine layers within a deposit) [21]
    nfx    - number of faults [0.3]
    ofx    - Starting position of faults (percentage of total model) [0.4]
    dx     - Spacing between faults (percentage of total model) [0.1]
    ofz    - Central depth of faults (percentage of total model) [0.3]
    rect   - radius for gaussian smoother [0.5]

  Returns:
    The velocity, reflectivity, fault label and image all of size [nx,nz]
  """
  # Model building object
  mb = mdlbuild.mdlbuild(nx,dx,ny=200,dy=dx,dz=dz,basevel=5000)
  nzi = 1000 # internal size is 1000

  # Propagation velocities
  props = np.linspace(maxvel,minvel,nlayer)

  # Specify the thicknesses
  thicks = np.random.randint(40,61,nlayer)

  dlyr = 0.05
  for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
    mb.deposit(velval=props[ilyr],thick=thicks[ilyr],dev_pos=0.0,layer=50,layer_rand=0.00,dev_layer=dlyr)

  # Water deposit
  mb.deposit(1480,thick=80,layer=150,dev_layer=0.0)

  # Trim model before faulting
  mb.trim(0,1100)

  # Put in the faults
  for ifl in progressbar(range(nfx), "nfaults:"):
    x = ofx + ifl*dfx
    mb.fault2d(begx=x,begz=ofz,daz=8000,dz=5000,azim=0.0,theta_die=11,theta_shift=4.0,dist_die=0.3,throwsc=10.0)

  # Get the model
  vel = gaussian_filter(mb.vel[:,:nzi].T,sigma=rect).astype('float32')
  lbl = mb.get_label2d()[:,:nzi].T
  ref = mb.get_refl2d()[:,:nzi].T
  # Parameters for ricker wavelet
  nt = 250; ot = 0.0; dt = 0.001; ns = int(nt/2)
  amp = 1.0; dly = 0.125
  minf = 100.0; maxf = 120.0
  # Create normalized image
  f = rndut.randfloat(minf,maxf)
  wav = ricker(nt,dt,f,amp,dly)
  img = dlut.normalize(np.array([np.convolve(ref[:,ix],wav) for ix in range(nx)])[:,ns:nzi+ns].T)
  nze = dlut.normalize(bandpass(np.random.rand(nzi,nx)*2-1, 2.0, 0.01, 2, pxd=43))/rndut.randfloat(3,5)
  img += nze

  # Window the models and return
  f1 = 50
  velwind = vel[f1:f1+nz,:]
  lblwind = lbl[f1:f1+nz,:]
  refwind = ref[f1:f1+nz,:]
  imgwind = img[f1:f1+nz,:]

  return velwind,refwind,imgwind,lblwind

def undulatingfaults2d(nz=512,nx=1000,dz=12.5,dx=25.0,nlayer=21,minvel=1600,maxvel=3000,rect=0.5,
                    nfx=3,ofx=0.4,dfx=0.1,ofz=0.3,noctaves=None,npts=None,amp=None):
  """
  Builds a 2D faulted velocity model with undulating layers
  Returns the velocity model, reflectivity, fault labels
  and a zero-offset image

  Parameters:
    nz       - number of depth samples [512]
    nx       - number of lateral samples[1000]
    dz       - depth sampling interval [12.5]
    dx       - lateral sampling interval [12.5]
    nlayer   - number of deposited layers (there exist many fine layers within a deposit) [21]
    nfx      - number of faults [0.3]
    ofx      - Starting position of faults (percentage of total model) [0.4]
    dx       - Spacing between faults (percentage of total model) [0.1]
    ofz      - Central depth of faults (percentage of total model) [0.3]
    rect     - radius for gaussian smoother [0.5]
    noctaves - octaves perlin parameters for squish [varies between 3 and 6]
    amp      - amplitude of folding [varies between 200 and 500]
    npts     - grid size for perlin noise [3]

  Returns:
    The velocity, reflectivity, fault label and image all of size [nx,nz]
  """
  # Model building object
  mb = mdlbuild.mdlbuild(nx,dx,ny=20,dy=dx,dz=dz,basevel=5000)
  nzi = 1000 # internal size is 1000

  # Propagation velocities
  props = np.linspace(maxvel,minvel,nlayer)

  # Specify the thicknesses
  thicks = np.random.randint(40,61,nlayer)

  dlyr = 0.05
  for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
    mb.deposit(velval=props[ilyr],thick=thicks[ilyr],dev_pos=0.0,layer=50,layer_rand=0.00,dev_layer=dlyr)
    if(ilyr == int(nlayer-2)):
      mb.squish(amp=300,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin',octaves=3,order=3)

  # Water deposit
  mb.deposit(1480,thick=80,layer=150,dev_layer=0.0)

  # Smooth the interface
  mb.smooth_model(rect1=1,rect2=5,rect3=1)

  # Trim model before faulting
  mb.trim(0,1100)

  # Put in the faults
  for ifl in progressbar(range(nfx), "nfaults:"):
    x = ofx + ifl*dfx
    mb.fault2d(begx=x,begz=ofz,daz=4000,dz=2500,azim=0.0,theta_die=11,theta_shift=6.0,dist_die=2.0,throwsc=10.0)
    mb.fault2d(begx=x,begz=ofz,daz=8000,dz=5000,azim=0.0,theta_die=11,theta_shift=4.0,dist_die=2.0,throwsc=10.0)

  # Get the model
  vel = gaussian_filter(mb.vel[:,:nzi].T,sigma=rect).astype('float32')
  lbl = mb.get_label2d()[:,:nzi].T
  ref = mb.get_refl2d()[:,:nzi].T
  # Parameters for ricker wavelet
  nt = 250; ot = 0.0; dt = 0.001; ns = int(nt/2)
  amp = 1.0; dly = 0.125
  minf = 100.0; maxf = 120.0
  # Create normalized image
  f = rndut.randfloat(minf,maxf)
  wav = ricker(nt,dt,f,amp,dly)
  img = dlut.normalize(np.array([np.convolve(ref[:,ix],wav) for ix in range(nx)])[:,ns:nzi+ns].T)
  nze = dlut.normalize(bandpass(np.random.rand(nzi,nx)*2-1, 2.0, 0.01, 2, pxd=43))/rndut.randfloat(3,5)
  img += nze

  # Window the models and return
  f1 = 50
  velwind = vel[f1:f1+nz,:]
  lblwind = lbl[f1:f1+nz,:]
  refwind = ref[f1:f1+nz,:]
  imgwind = img[f1:f1+nz,:]

  return velwind,refwind,imgwind,lblwind

def undulatingfaults_old(nz=512,nx=1000,dz=12.5,dx=25.0,nlayer=21,minvel=1600,maxvel=3000,rect=0.5,
                    nfx=3,ofx=0.4,dfx=0.1,ofz=0.3,noctaves=None,npts=None,amp=None):
  """
  Builds a 2D faulted velocity model with undulating layers
  Returns the velocity model, reflectivity, fault labels
  and a zero-offset image

  Parameters:
    nz       - number of depth samples [512]
    nx       - number of lateral samples[1000]
    dz       - depth sampling interval [12.5]
    dx       - lateral sampling interval [12.5]
    nlayer   - number of deposited layers (there exist many fine layers within a deposit) [21]
    nfx      - number of faults [0.3]
    ofx      - Starting position of faults (percentage of total model) [0.4]
    dx       - Spacing between faults (percentage of total model) [0.1]
    ofz      - Central depth of faults (percentage of total model) [0.3]
    rect     - radius for gaussian smoother [0.5]
    noctaves - octaves perlin parameters for squish [varies between 3 and 6]
    amp      - amplitude of folding [varies between 200 and 500]
    npts     - grid size for perlin noise [3]

  Returns:
    The velocity, reflectivity, fault label and image all of size [nx,nz]
  """
  # Model building object
  mb = mdlbuild.mdlbuild(nx,dx,ny=200,dy=dx,dz=dz,basevel=5000)
  nzi = 1000 # internal size is 1000

  # Propagation velocities
  props = np.linspace(maxvel,minvel,nlayer)

  # Specify the thicknesses
  thicks = np.random.randint(40,61,nlayer)

  dlyr = 0.05
  for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
    mb.deposit(velval=props[ilyr],thick=thicks[ilyr],dev_pos=0.0,layer=50,layer_rand=0.00,dev_layer=dlyr)
    if(ilyr == int(nlayer-2)):
      mb.squish(amp=300,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin',octaves=3,order=0)

  # Water deposit
  mb.deposit(1480,thick=80,layer=150,dev_layer=0.0)

  # Smooth the interface
  mb.smooth_model(rect1=1,rect2=5,rect3=1)

  # Trim model before faulting
  mb.trim(0,1100)

  # Put in the faults
  for ifl in progressbar(range(nfx), "nfaults:"):
    x = ofx + ifl*dfx
    mb.fault(begx=x,begz=ofz,daz=4000,dz=2500,azim=0.0,theta_die=11,theta_shift=6.0,dist_die=2.0,throwsc=10.0)
    mb.fault(begx=x,begz=ofz,daz=8000,dz=5000,azim=0.0,theta_die=11,theta_shift=4.0,dist_die=2.0,throwsc=10.0)

  # Get the model
  vel = gaussian_filter(mb.vel[100,:,:nzi].T,sigma=rect).astype('float32')
  lbl = mb.get_label()[100,:,:nzi].T
  ref = mb.get_refl()[100,:,:nzi].T
  # Parameters for ricker wavelet
  nt = 250; ot = 0.0; dt = 0.001; ns = int(nt/2)
  amp = 1.0; dly = 0.125
  minf = 30.0; maxf = 60.0
  # Create normalized image
  f = rndut.randfloat(minf,maxf)
  wav = ricker(nt,dt,f,amp,dly)
  img = dlut.normalize(np.array([np.convolve(ref[:,ix],wav) for ix in range(nx)])[:,ns:nzi+ns].T)
  nze = dlut.normalize(bandpass(np.random.rand(nzi,nx)*2-1, 2.0, 0.01, 2, pxd=43))/rndut.randfloat(3,5)
  img += nze

  # Window the models and return
  f1 = 50
  velwind = vel[f1:f1+nz,:]
  lblwind = lbl[f1:f1+nz,:]
  refwind = ref[f1:f1+nz,:]
  imgwind = img[f1:f1+nz,:]

  return velwind,refwind,imgwind,lblwind

def undulatingrandfaults2d(nz=512,nx=1000,dz=12.5,dx=25.0,nlayer=21,minvel=1600,maxvel=3000,rect=0.5,
                    nfx=3,ofx=0.4,dfx=0.1,ofz=0.3,noctaves=None,npts=None,amp=None):
  """
  Builds a 2D faulted velocity model with undulating layers
  Returns the velocity model, reflectivity, fault labels
  and a zero-offset image

  Parameters:
    nz       - number of depth samples [512]
    nx       - number of lateral samples[1000]
    dz       - depth sampling interval [12.5]
    dx       - lateral sampling interval [12.5]
    nlayer   - number of deposited layers (there exist many fine layers within a deposit) [21]
    nfx      - number of faults [0.3]
    ofx      - Starting position of faults (percentage of total model) [0.4]
    dx       - Spacing between faults (percentage of total model) [0.1]
    ofz      - Central depth of faults (percentage of total model) [0.3]
    rect     - radius for gaussian smoother [0.5]
    noctaves - octaves perlin parameters for squish [varies between 3 and 6]
    amp      - amplitude of folding [varies between 200 and 500]
    npts     - grid size for perlin noise [3]

  Returns:
    The velocity, reflectivity, fault label and image all of size [nx,nz]
  """
  # Model building object
  # Remember to change dist_die based on ny
  mb = mdlbuild.mdlbuild(nx,dx,ny=20,dy=dx,dz=dz,basevel=5000)
  nzi = 1000 # internal size is 1000

  # Propagation velocities
  props = np.linspace(maxvel,minvel,nlayer)

  # Specify the thicknesses
  thicks = np.random.randint(40,61,nlayer)

  dlyr = 0.05
  for ilyr in progressbar(range(nlayer), "ndeposit:", 40):
    mb.deposit(velval=props[ilyr],thick=thicks[ilyr],dev_pos=0.0,layer=50,layer_rand=0.00,dev_layer=dlyr)
    if(ilyr == int(nlayer-2)):
      amp  = rndut.randfloat(200,500)
      octs = np.random.randint(2,7)
      npts = np.random.randint(2,5)
      mb.squish(amp=amp,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin',npts=npts,octaves=octs,order=3)

  # Water deposit
  mb.deposit(1480,thick=80,layer=150,dev_layer=0.0)

  # Smooth the interface
  mb.smooth_model(rect1=1,rect2=5,rect3=1)

  # Trim model before faulting
  mb.trim(0,1100)

  #XXX: Thresh should be a function of theta_shift

  # Generate the fault positions
  flttype = np.random.choice([0,1,2,3,4,5])

  if(flttype == 0):
    largefaultblock(mb,0.3,0.7,ofz,nfl=6)
  elif(flttype == 1):
    slidingfaultblock(mb,0.3,0.7,ofz,nfl=6)
  elif(flttype == 2):
    mediumfaultblock(mb,0.3,0.7,0.25,space=0.02,nfl=10)
  elif(flttype == 3):
    mediumfaultblock(mb,0.3,0.7,0.25,space=0.005,nfl=20)
  elif(flttype == 4):
    tinyfaultblock(mb,0.3,0.7,0.25,space=0.02,nfl=10)
  else:
    tinyfaultblock(mb,0.3,0.7,0.25,space=0.005,nfl=20)

  # Get the model
  vel = gaussian_filter(mb.vel[:,:nzi].T,sigma=rect).astype('float32')
  lbl = mb.get_label2d()[:,:nzi].T
  ref = mb.get_refl2d()[:,:nzi].T
  # Parameters for ricker wavelet
  nt = 250; ot = 0.0; dt = 0.001; ns = int(nt/2)
  amp = 1.0; dly = 0.125
  minf = 100.0; maxf = 120.0
  # Create normalized image
  f = rndut.randfloat(minf,maxf)
  wav = ricker(nt,dt,f,amp,dly)
  img = dlut.normalize(np.array([np.convolve(ref[:,ix],wav) for ix in range(nx)])[:,ns:nzi+ns].T)
  nze = dlut.normalize(bandpass(np.random.rand(nzi,nx)*2-1, 2.0, 0.01, 2, pxd=43))/rndut.randfloat(3,5)
  img += nze

  # Window the models and return
  f1 = 50
  velwind = vel[f1:f1+nz,:]
  lblwind = lbl[f1:f1+nz,:]
  refwind = ref[f1:f1+nz,:]
  imgwind = img[f1:f1+nz,:]

  return velwind,refwind,imgwind,lblwind

#TODO: add some randomness to the daz,dz and theta_shift
def largefaultblock(mb,minpos,maxpos,ofz,nfl=6):
  lfpars = mb.getfaultpos2d(minpos,maxpos,minblk=0.01,minhor=0.08,mingrb=0.05,nfaults=nfl)
  for ifl in progressbar(range(nfl), "nlfaults:"):
    iflpar = lfpars[ifl]
    z = ofz + rndut.randfloat(-0.01,0.01)
    daz = 8000 + rndut.randfloat(-1000,1000)
    dz  = 5000 + rndut.randfloat(-1000,1000)
    theta_shift = 3.0 + rndut.randfloat(-1.0,1.0)
    mb.fault2d(begx=iflpar[0],begz=z,daz=daz,dz=dz,azim=iflpar[1],theta_die=11,theta_shift=theta_shift,dist_die=2.0,throwsc=10.0,thresh=0.1)

def slidingfaultblock(mb,minpos,maxpos,ofz,nfl=6):
  sfpars = mb.getfaultpos2d(minpos,maxpos,minblk=0.02,minhor=0.2,mingrb=0.2,nfaults=nfl)
  for ifl in progressbar(range(nfl), "nsfaults:"):
    ifspar = sfpars[ifl]
    z = ofz + rndut.randfloat(-0.01,0.01)
    daz = 7500  + rndut.randfloat(-1000,1000)
    dz  = 15000 + rndut.randfloat(-1000,1000)
    theta_shift = 3.0 + rndut.randfloat(-1.0,1.0)
    mb.fault2d(begx=ifspar[0],begz=ofz,daz=daz,dz=dz,azim=ifspar[1],theta_die=11,theta_shift=theta_shift,dist_die=2.0,throwsc=10.0,thresh=0.1)

def mediumfaultblock(mb,minpos,maxpos,ofz,space,nfl=10):
  mfpars = mb.getfaultpos2d(minpos,maxpos,minblk=space,minhor=space,mingrb=space,nfaults=nfl)
  for ifl in progressbar(range(nfl), "nmfaults:"):
    ifmpar = mfpars[ifl]
    z = ofz + rndut.randfloat(-0.01,0.01)
    daz = 4000 + rndut.randfloat(-1000,1000)
    dz  = 2500 + rndut.randfloat(-1000,1000)
    theta_shift = 3.0 + rndut.randfloat(-1.0,1.0)
    mb.fault2d(begx=ifmpar[0],begz=z,daz=daz,dz=dz,azim=ifmpar[1],theta_die=11,theta_shift=theta_shift,dist_die=2.0,throwsc=10.0,thresh=0.2)

def tinyfaultblock(mb,minpos,maxpos,ofz,space,nfl=10):
  tfpars = mb.getfaultpos2d(minpos,maxpos,space,space,space,nfaults=nfl)
  for ifl in progressbar(range(nfl), "ntfaults:"):
    iftpar = tfpars[ifl]
    z = ofz + rndut.randfloat(-0.01,0.01)
    daz = 2000 + rndut.randfloat(-500,500)
    dz  = 1250 + rndut.randfloat(-250,250)
    theta_shift = 4.0 + rndut.randfloat(-1.0,1.0)
    mb.fault2d(begx=iftpar[0],begz=z,daz=daz,dz=dz,azim=iftpar[1],theta_die=11,theta_shift=theta_shift,dist_die=2.0,throwsc=10.0,thresh=0.15)

