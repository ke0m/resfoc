"""
Functions that create custom velocity models
suited for particular problems

@author: Joseph Jennings
@version: 2020.10.23
"""
import numpy as np
import velocity.mdlbuild as mdlbuild
from deeplearn.utils import resample, normalize
from oway.utils import interp_vel
from oway.costaper import costaper
from genutils.rand import randfloat
from genutils.plot import plot_img2d

def random_hale_vel(nz=900,nx=800,dz=0.005,dx=0.01675,vzin=None):
  """
  Generates a random realization of the Hale/BEI
  velocity model

  Parameters:
    nz   - output number of depth samples [900]
    nx   - output number of lateral samples [800]
    dz   - depth sampling [0.005]
    dx   - lateral sampling [0.01675]
    vzin - a vzin that determines the velocity values [None]
  """
  dzm,dxm  = dz*1000,dx*1000
  nlayer = 200
  minvel,maxvel = 1600,5000
  vz = np.linspace(maxvel,minvel,nlayer)
  if(vzin is not None):
    vzr = resample(vzin,90)*1000
    vz[-90:] = vzr[::-1]

  mb = mdlbuild.mdlbuild(nx,dxm,20,dy=dxm,dz=dzm,basevel=5000)

  thicks = np.random.randint(5,15,nlayer)

  # Randomize the squishing depth
  sqz = np.random.choice(list(range(180,199)))

  dlyr = 0.05
  # Build the sedimentary layers
  for ilyr in range(nlayer):
    mb.deposit(velval=vz[ilyr],thick=thicks[ilyr],dev_pos=0.0,
               layer=50,layer_rand=0.00,dev_layer=dlyr)
    if(ilyr == sqz):
      mb.squish(amp=150,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin',octaves=3,order=3)

  mb.deposit(1480,thick=40,layer=150,dev_layer=0.0)
  mb.trim(top=0,bot=nz)

  # Pos
  xpos = np.asarray([0.25,0.30,0.432,0.544,0.6,0.663])
  xhi = xpos + 0.04
  xlo = xpos - 0.04
  cxpos = np.zeros(xpos.shape)

  nflt = len(xpos)
  for iflt in range(nflt):
    cxpos[iflt] = randfloat(xlo[iflt],xhi[iflt])
    if(iflt > 0 and cxpos[iflt] - cxpos[iflt-1] < 0.07):
      cxpos[iflt] += 0.07
    cdaz = randfloat(16000,20000)
    cdz = cdaz + randfloat(0,6000)
    # Choose the theta_die
    theta_die = randfloat(1.5,3.5)
    if(theta_die < 2.7):
      begz = randfloat(0.23,0.26)
    else:
      begz = randfloat(0.26,0.33)
    fpr = np.random.choice([True,True,False])
    rd = randfloat(52,65)
    dec = randfloat(0.94,0.96)
    mb.fault2d(begx=cxpos[iflt],begz=begz,daz=cdaz,dz=cdz,azim=180,theta_die=theta_die,theta_shift=4.0,dist_die=2.0,
               throwsc=35.0,fpr=fpr,rectdecay=rd,dec=dec)
  velw = mb.vel
  refw = normalize(mb.get_refl2d())
  lblw = mb.get_label2d()

  return velw*0.001,refw,lblw

def fake_fault_img(vel,img,ox=7.035,dx=0.01675,ovx=7.035,dvx=0.0335,dz=0.005):
  """
  Puts a fake fault in the Hale/BEI image
  and prepares for the application of the Hessian

  Parameters:
    img - the migrated Hale/BEI image [nx,nz]
  """
  nx,nz = img.shape
  nvx,nvz = vel.shape

  # Taper the image
  img  = np.ascontiguousarray(img).astype('float32')[20:-20,:]
  imgt = costaper(img,nw2=60)

  # Pad the image
  imgp = np.pad(imgt,((110,130),(0,0)),mode='constant')

  # Replicate the image to make it 2.5D
  imgp3d = np.repeat(imgp[np.newaxis],20,axis=0)

  veli = vel[np.newaxis] #[ny,nx,nz]
  veli = np.ascontiguousarray(np.transpose(veli,(2,0,1))) # [ny,nx,nz] -> [nz,ny,nx]

  # Interpolate the velocity model
  veli = interp_vel(nz,
                    1,0.0,1.0,
                    nx,ox,dx,
                    veli,dvx,1.0,ovx,0.0)
  veli = veli[:,0,:].T

  velp = np.pad(veli,((90,110),(0,0)),mode='edge')

  # Build a model that is the same size
  minvel = 1600; maxvel = 5000
  nlayer = 200
  dzm,dxm  = dz*1000,dx*1000
  nzm,nxm = nz,800
  mb = mdlbuild.mdlbuild(nxm,dxm,20,dy=dxm,dz=dzm,basevel=5000)
  props = mb.vofz(nlayer,minvel,maxvel,npts=2)
  thicks = np.random.randint(5,15,nlayer)

  dlyr = 0.05
  for ilyr in range(nlayer):
    mb.deposit(velval=props[ilyr],thick=thicks[ilyr],dev_pos=0.0,
               layer=50,layer_rand=0.00,dev_layer=dlyr)

  mb.trim(top=0,bot=900)

  mb.vel[:] = imgp3d[:]
  mb.fault2d(begx=0.7,begz=0.26,daz=20000,dz=24000,azim=180.0,theta_die=2.5,theta_shift=4.0,dist_die=2.0,
             throwsc=35.0,fpr=False)

  refw = normalize(mb.vel)
  lblw = mb.get_label2d()

  return velp,refw,lblw

