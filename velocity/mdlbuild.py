import numpy as np
import velocity.evntcre8 as evntcre8
from utils.ptyprint import progressbar,printprogress
import scaas.noise_generator as noise_generator
from scaas.gradtaper import build_taper_ds
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

class mdlbuild:
  """ 
  Builds random geologically feasible velocity models.
  Based on the syntheticModel code from Bob Clapp

  @author: Joseph Jennings
  @version: 2020.02.05
  """

  def __init__(self,nx,dx,ny,dy,dz,nbase=50,basevel=4000):
    # Get dimensions
    self.__nbase = nbase; self.__nx = nx; self.__ny = ny
    # Get samplings
    self.__dz = dz; self.__dx = dx; self.__dy = dy
    # Create the basement layer
    self.vel = np.zeros([self.__ny,self.__nx,nbase],dtype='float32')
    self.lyr = np.zeros([self.__ny,self.__nx,nbase],dtype='int32')
    self.lbl = np.zeros([self.__ny,self.__nx,nbase],dtype='float32')
    self.vel[:] = basevel; self.lyr[:] = 0
    self.enum = 1 # Geological event number
    # Event creating object
    self.ec8 = evntcre8.evntcre8(self.__nx,self.__ny,self.__dx,self.__dy,self.__dz)

  def deposit(self,velval=1400,thick=30,band1=0.4,band2=0.02,band3=0.02,dev_pos=0.0,
              layer=23,layer_rand=0.3,dev_layer=0.26):
    """
    Creates a deposition event in the geological model. 
    
    Parameters:
      A general summary of the parameters. 
      
      Lateral variation:
      The band1-3 parameters basically are three parameters used
      to define a 3D bandpass filter which is applied to a cube of random numbers. The different
      bands allow for different amounts of variation in each direction. These and dev_pos control
      the variation of the deposit in all dimensions. Setting dev_pos to 0 will result in no
      lateral variation and the code will run much faster

      Vertical variation:
      The parameters layer,layer_rand and dev_layer dertermine the variation of the layering
      within a unit. Thick and dev_layer have the most impact where layer_rand is more of a 
      fine tuning parameter
      
      velval     - base value of velocity in the layer in m/s [1400]
      thick      - thickness of the layer in samples [30]
      band1      - bandpass parameter for axis 1 [0.4]
      band2      - bandpass parameter for axis 2 [0.02]
      band3      - bandpass parameter for axis 3 [0.02]
      dev_pos    - Determines the strength of the varation of the velocity in all directions by
                   bandpassing randmom numbers (setting to 0 will make it much faster) [0.0]
      layer      - Changes the thickness of the thin layering (bigger number is thicker) [23]
      layer_rand - Also change the thickness of the thin layering [0.0]. A larger number leads
                   to more thin bedding within a layer
      dev_layer  - Changes the variation of the velocity within a layer. Larger values lead to more variation
                   Set to 0 will result in homogeonous layer [0.26]
    """
    # First, expand the model
    nzin = self.vel.shape[2]
    nzot = int(nzin + thick)
    velot = np.zeros([self.__ny,self.__nx,nzot],dtype='float32')
    lyrot = np.zeros([self.__ny,self.__nx,nzot],dtype='int32')
    self.ec8.expand(thick,0,nzin,self.lyr,self.vel,nzot,lyrot,velot)
    # Next create the deposit
    self.ec8.deposit(velval,
                     band1,band2,band3,
                     layer,layer_rand,dev_layer,dev_pos,
                     nzot,lyrot,velot)
    # Update the layer and velocity models
    self.vel = velot
    self.lyr = lyrot

  def vofz(self,nlayer=20,minvel=1600,maxvel=5000,npts=2,octaves=3,persist=0.3):
    """ Generate a random v(z) that defines propagation velocities """
    props = np.zeros(nlayer)
    # Make sure we do not go out of bounds
    while(np.min(props) != minvel or np.max(props) != maxvel):
      props = np.linspace(maxvel,minvel,nlayer)
      ptb = noise_generator.perlin(x=np.linspace(0,npts,nlayer), octaves=octaves, period=80, Ngrad=80, persist=persist, ncpu=1)
      ptb -= np.mean(ptb);
      # Define taper so that the ends are not perturbed
      tap,_ = build_taper_ds(1,nlayer,1,5,15,19)
      props += 5000*(ptb*tap)

    return props

  def fault(self,begx=0.5,begy=0.5,begz=0.5,daz=8000,dz=7000,azim=180,
      theta_die=12,theta_shift=4.0,dist_die=0.3,perp_die=0.5,dirf=0.1,throwsc=1.0,thresh=50):
    """
    Creates a fault event in the geologic model

    Parameters:
      azim        - Azimuth of fault [180]
      begx        - Relative location of the beginning of the fault in x [0.5]
      begy        - Relative location of the beginning of the fault in y [0.5]
      begz        - Relative location of the beginning of the fault in z [0.5]
      dz          - Distance away from the center of a circle in z [7000]
      daz         - Distance away in azimuth [8000]
      perp_die    - Controls the die off perpendicular to the fault
                    (e.g., if fault is visible in x, die off is in y).
                    Large number results in slower dieoff [0.5]
      dist_die    - Controls the die off in the same plane of the fault 
                    (e.g., if fault is visible in x, die off is also in x)
                    Large number results in slower dieoff [0.3]
      theta_die   - Controls the die off along the fault (essentially the throw). The larger 
                    the number the larger the fault will be. Acts similar to daz. [12]
      theta_shift - Shift in theta for fault [4.0]
      dirf        - Direction of fault movement [0.1]
      throwsc     - The total shift in z is divided by this amount (leads to smaller throw) [1.0]
      thresh      - Threshold applied for obtaining the fault labels [50]
    """
    # Create the output velocity, layermodel and label
    nz = self.vel.shape[2]
    velot = np.zeros(self.vel.shape,dtype='float32')
    lyrot = np.zeros(self.lyr.shape,dtype='int32')
    lblto = np.zeros(self.vel.shape,dtype='float32')
    lbltn = np.zeros(self.vel.shape,dtype='float32')
    lblot = np.zeros(self.vel.shape,dtype='float32')
    if(self.lbl.shape[2] != lblot.shape[2]):
      ndiff = lblot.shape[2] - self.lbl.shape[2]
      self.lbl = np.pad(self.lbl,((0,0),(0,0),(ndiff,0)),'constant')
    # Create the fault
    self.ec8.fault(nz,self.lyr,self.vel,self.lbl,
                   azim,begx,begy,begz,dz,daz,
                   theta_shift,perp_die,dist_die,theta_die,dirf,throwsc,
                   lyrot,velot,lblto,lbltn)
    # Update old label with shifted version
    self.lbl = lblto
    # Update layer and velocity models
    self.vel = velot
    self.lyr = lyrot
    # Compute laplacian of output shift map
    self.ec8.laplacian(nz,lbltn,lblot)
    # Apply a threshold
    idx = np.abs(lblot) > thresh
    lblot[ idx] = 1; lblot[~idx] = 0
    # Update label
    self.lbl += lblot
    # Apply final threshold to label
    idx = self.lbl > 1
    self.lbl[idx] = 1

  def get_label(self):
    """
    Gets the fault labels and ensures the label is the same size
    as the current velocity model size
    """
    nzv = self.vel.shape[2]
    nzl = self.lbl.shape[2]
    if(nzv == nzl):
      return self.lbl
    else:
      ndiff = nzv - nzl
      return np.pad(self.lbl,((0,0),(0,0),(ndiff,0)),'constant')

  def tinyfault_block(self,nfault=5,azim=0.0,begz=0.2,begx=0.3,begy=0.3,dx=0.1,dy=0.0,rand=True):
    """
    Puts in a tiny fault block system. For now, only will give nice faults along
    0,90,180,270 azimuths

    Parameters:
      nfault - number of faults in the system [5]
      azim   - azimuth along which faults are oriented [0.0]
      begz   - beginning position in z for fault (same for all) [0.3]
      begx   - beginning position in x for system [0.5]
      begy   - beginning position in y for system [0.5]
      dx     - spacing between faults in the x direction [0.1]
      dy     - spacing between faults in the y direction [0.0]
      rand   - small random variations in the positioning and throw of the faults [True]
    """
    signx = 1; signy = 1
    if(begx > 0.5):
      signx = -1
    if(begy > 0.5):
      signy = -1
    for ifl in progressbar(range(nfault), "ntfaults:", 40):
      daz = 3000; dz = 3000; dxi = dx; dyi = dy
      if(rand):
        daz += np.random.rand()*(2*1000) - 1000.0
        dz  += np.random.rand()*(2*500)  - 500.0
        dxi += np.random.rand()*dxi - dxi/2.0
        dyi += np.random.rand()*dyi - dyi/2.0
      self.fault(begx=begx,begy=begy,begz=begz,daz=daz,dz=dz,azim=azim,theta_die=9.0,theta_shift=4.0,dist_die=0.3,perp_die=1.0)
      # Move along x or y
      begx += signx*dxi; begy += signy*dyi

  def smallfault_block(self,nfault=5,azim=0.0,begz=0.3,begx=0.3,begy=0.3,dx=0.1,dy=0.0,rand=True):
    """
    Puts in a small fault block system. For now, only will give nice faults along
    0,90,180,270 azimuths

    Parameters:
      nfault - number of faults in the system [5]
      azim   - azimuth along which faults are oriented [0.0]
      begz   - beginning position in z for fault (same for all) [0.3]
      begx   - beginning position in x for system [0.5]
      begy   - beginning position in y for system [0.5]
      dx     - spacing between faults in the x direction [0.1]
      dy     - spacing between faults in the y direction [0.0]
      rand   - small random variations in the positioning and throw of the faults [True]
    """
    signx = 1; signy = 1
    if(begx > 0.5):
      signx = -1
    if(begy > 0.5):
      signy = -1
    for ifl in progressbar(range(nfault), "nsfaults:", 40):
      daz = 8000; dz = 5000; dxi = dx; dyi = dy
      if(rand):
        daz += np.random.rand()*(2000) - 1000
        dz  += np.random.rand()*(2000) - 1000
        dxi += np.random.rand()*dxi - dxi/2
        dyi += np.random.rand()*dyi - dyi/2
      self.fault(begx=begx,begy=begy,begz=begz,daz=daz,dz=dz,azim=azim,theta_die=11.0,theta_shift=4.0,dist_die=0.3,perp_die=1.0)
      # Move along x or y
      begx += signx*dxi; begy += signy*dyi

  def largefault_block(self,nfault=3,azim=0.0,begz=0.6,begx=0.5,begy=0.5,dx=0.2,dy=0.0,rand=True):
    """
    Puts in a large fault block system. For now, only will give nice faults along
    0,90,180,270 azimuths

    Parameters:
      nfault - number of faults in the system [3]
      azim   - azimuth along which faults are oriented [0.0]
      begz   - beginning position in z for fault (same for all) [0.6]
      begx   - beginning position in x for system [0.5]
      begy   - beginning position in y for system [0.5]
      dx     - spacing between faults in the x direction [0.2]
      dy     - spacing between faults in the y direction [0.0]
      rand   - small random variations in the positioning and throw of the faults [True]
    """
    signx = 1; signy = 1
    if(begx > 0.5):
      signx = -1
    if(begy > 0.5):
      signy = -1
    for ifl in progressbar(range(nfault), "nlfaults:", 40):
      daz = 25000; dz = 10000; dxi = dx; dyi = dy
      if(rand):
        daz += np.random.rand()*(2000) - 1000
        dz  += np.random.rand()*(2000) - 1000
        dxi += np.random.rand()*dxi - dxi/2
        dyi += np.random.rand()*dyi - dyi/2
      self.fault(begx=begx,begy=begy,begz=begz,daz=daz,dz=dz,azim=azim,theta_die=12.0,theta_shift=4.0,dist_die=1.5,perp_die=1.0,thresh=200)
      # Move along x or y
      begx += signx*dxi; begy += signy*dyi

  def sliding_block(self,nfault=5,azim=0.0,begz=0.5,begx=0.5,begy=0.5,dx=0.2,dy=0.0,rand=True):
    """
    Puts in sliding fault block system. For now, only will give nice faults along
    0,90,180,270 azimuths

    Parameters:
      nfault - number of faults in the system [5]
      azim   - azimuth along which faults are oriented [0.0]
      begz   - beginning position in z for fault (same for all) [0.6]
      begx   - beginning position in x for system [0.5]
      begy   - beginning position in y for system [0.5]
      dx     - spacing between faults in the x direction [0.2]
      dy     - spacing between faults in the y direction [0.0]
      rand   - small random variations in the positioning and throw of the faults [True]
    """
    signx = 1; signy = 1
    if(begx > 0.5):
      signx = -1
    if(begy > 0.5):
      signy = -1
    for ifl in progressbar(range(nfault), "ndfaults:", 40):
      daz = 10000; dz = 25000; dxi = dx; dyi = dy
      if(rand):
        daz += np.random.rand()*(2000) - 1000
        dz  += np.random.rand()*(2000) - 1000
        dxi += np.random.rand()*(dxi) - dxi/2
        dyi += np.random.rand()*(dyi) - dyi/2
      self.fault(begx=begx,begy=begy,begz=begz,daz=daz,dz=dz,azim=azim,theta_die=12.0,theta_shift=4.0,dist_die=1.5,perp_die=1.0,thresh=200)
      # Move along x or y
      begx += signx*dxi; begy += signy*dyi

  def smallgraben_block(self,azim=0.0,begz=0.5,begx=0.5,begy=0.5,dx=0.1,dy=0.0,rand=True):
    """
    Puts in a small graben fault block system. For now only will give nice faults along
    0,90,180,270 azimuths

    Parameters
      azim - azimuth along which faults are oriented [0.0]
      begz   - beginning position in z for fault (same for all) [0.6]
      begx   - beginning position in x for system [0.5]
      begy   - beginning position in y for system [0.5]
      dx     - spacing between faults in the x direction [0.1]
      dy     - spacing between faults in the y direction [0.0]
      rand   - small random variations in the positioning and throw of the faults [True]
    """
    assert(dx != 0.0 or dy != 0.0),"Either dx or dy must be non-zero"
    # Throw parameters and spacing
    daz1 = 6000; dz1 = 3000; dxi = dx; dyi = dy
    daz2 = 6000; dz2 = 3000;
    if(rand):
      # First fault
      daz1 += np.random.rand()*(2000) - 1000
      dz1  += np.random.rand()*(1000) - 500
      # Second fault
      daz2 += np.random.rand()*(2000) - 1000
      dz2  += np.random.rand()*(1000) - 500
      # Spacing
      dxi += np.random.rand()*(dxi) - dxi/2
      dyi += np.random.rand()*(dyi) - dyi/2
    if(dx != 0.0):
      # First fault
      printprogress("nsfaults",0,2)
      self.fault(begx=begx    ,begy=begy,begz=begz,daz=daz1,dz=dz1,azim=azim+180.0,theta_die=12.0,theta_shift=4.0,dist_die=1.2,perp_die=1.0)
      printprogress("nsfaults",1,2)
      # Second fault
      self.fault(begx=begx+dx,begy=begy,begz=begz,daz=daz2,dz=dz2,azim=azim       ,theta_die=12.0,theta_shift=4.0,dist_die=1.2,perp_die=1.0)
      printprogress("nsfaults",2,2)
    else:
      # First fault
      printprogress("nsfaults",0,2)
      self.fault(begx=begx,begy=begy    ,begz=begz,daz=daz1,dz=dz1,azim=azim+180.0,theta_die=12.0,theta_shift=4.0,dist_die=1.2,perp_die=1.0)
      printprogress("nsfaults",1,2)
      # Second fault
      self.fault(begx=begx,begy=begy+dy,begz=begz,daz=daz2,dz=dz2,azim=azim      ,theta_die=12.0,theta_shift=4.0,dist_die=1.2,perp_die=1.0)
      printprogress("nsfaults",2,2)

  def largegraben_block(self,azim=0.0,begz=0.6,begx=0.3,begy=0.5,dx=0.3,dy=0.0,rand=True):
    """
    Puts in a large graben fault block system. For now only will give nice faults along
    0,90,180,270 azimuths

    Parameters
      azim - azimuth along which faults are oriented [0.0]
      begz - beginning position in z for fault (same for all) [0.6]
      begx - beginning position in x for system [0.5]
      begy - beginning position in y for system [0.5]
      dx   - spacing between faults in the x direction [0.3]
      dy   - spacing between faults in the y direction [0.0]
      rand - small random variations in the positioning and throw of the faults [True]
    """
    assert(dx != 0.0 or dy != 0.0),"Either dx or dy must be non-zero"
    # Throw parameters and spacing
    daz1 = 25000; dz1 = 10000; dxi = dx; dyi = dy
    daz2 = 25000; dz2 = 10000
    if(rand):
      # First fault
      daz1 += np.random.rand()*(2000) - 1000
      dz1  += np.random.rand()*(2000) - 1000
      # Second fault
      daz2 += np.random.rand()*(2000) - 1000
      dz2  += np.random.rand()*(2000) - 1000
      # Spacing
      dxi += np.random.rand()*(dxi) - dxi/2
      dyi += np.random.rand()*(dyi) - dyi/2
    if(dx != 0.0):
      # First fault
      printprogress("nlfaults",0,2)
      self.fault(begx=begx    ,begy=begy,begz=begz,daz=daz1,dz=dz1,azim=azim+180.0,theta_die=12.0,theta_shift=4.0,dist_die=1.2,perp_die=1.0)
      printprogress("nlfaults",1,2)
      # Second fault
      self.fault(begx=begx+dx,begy=begy,begz=begz,daz=daz2,dz=dz2,azim=azim      ,theta_die=12.0,theta_shift=4.0,dist_die=1.2,perp_die=1.0)
      printprogress("nlfaults",2,2)
    else:
      # First fault
      printprogress("nlfaults",0,2)
      self.fault(begx=begx,begy=begy    ,begz=begz,daz=daz1,dz=dz1,azim=azim+180.0,theta_die=12.0,theta_shift=4.0,dist_die=1.2,perp_die=1.0)
      printprogress("nlfaults",1,2)
      # Second fault
      self.fault(begx=begx,begy=begy+dy,begz=begz,daz=daz,dz=dz,azim=azim      ,theta_die=12.0,theta_shift=4.0,dist_die=1.2,perp_die=1.0)
      printprogress("nlfaults",2,2)

  def smallhorstgraben_block(self,azim=0.0,begz=0.5,rand=True,xdir=True):
    """
    Puts in a small horst-graben block fault system.
    Attempts to span most of the lateral range of the model.
    For now, will only give nice faults along 0,90,180,270 azimuths

    Parameters:
      azim - azimuth along which faults are oriented [0.0]
      begz - beginning position in z for fault (same for all) [0.5]
      xdir - Whether the faults should be spaced along the x direction [True]
      rand - small random variations in the throw of the faults [True]
    """
    dx   = 0.0; dy   = 0.0
    begx = 0.5; begy = 0.5
    if(xdir):
      dx = 0.16; begx = 0.05
    else:
      dy = 0.16; begy = 0.05
    for ifl in progressbar(range(6), "ngrabens:", 40):
      daz = 5000; dz = 3000
      if(rand):
        daz += np.random.rand()*(2000) - 1000
        dz  += np.random.rand()*(2000) - 1000
      # Put in graben pair along x or y
      if(xdir):
        self.fault(begx=begx     ,begy=begy,begz=begz,daz=daz,dz=dz,azim=azim+180.0,theta_die=12.0,theta_shift=4.0,dist_die=1.5,perp_die=1.0,thresh=200)
        self.fault(begx=begx+0.07,begy=begy,begz=begz,daz=daz,dz=dz,azim=azim      ,theta_die=12.0,theta_shift=4.0,dist_die=1.5,perp_die=1.0,thresh=200)
      else:
        self.fault(begx=begx,begy=begy     ,begz=begz,daz=daz,dz=dz,azim=azim+180.0,theta_die=12.0,theta_shift=4.0,dist_die=1.5,perp_die=1.0,thresh=200)
        self.fault(begx=begx,begy=begy+0.07,begz=begz,daz=daz,dz=dz,azim=azim      ,theta_die=12.0,theta_shift=4.0,dist_die=1.5,perp_die=1.0,thresh=200)
      # Move along x or y
      begx += dx; begy += dy

  def largehorstgraben_block(self,azim=0.0,begz=0.1,xdir=True,rand=True):
    """
    Puts in a small horst-graben block fault system.
    Attempts to span most of the lateral range of the model.
    For now, will only give nice faults along 0,90,180,270 azimuths

    Parameters:
      azim - azimuth along which faults are oriented [0.0]
      begz - beginning position in z for fault (same for all) [0.5]
      xdir - Whether the faults should be spaced along the x direction [True]
      rand - small random variations in the throw of the faults [True]
    """
    dx   = 0.0; dy   = 0.0
    begx = 0.5; begy = 0.5
    if(xdir):
      dx = 0.32; begx = 0.05
    else:
      dy = 0.32; begy = 0.05
    for ifl in progressbar(range(3), "ngrabens:", 40):
      daz = 15000; dz = 5000
      if(rand):
        daz += np.random.rand()*(2000) - 1000
        dz  += np.random.rand()*(2000) - 1000
      # Put in graben pair along x or y
      if(xdir):
        self.fault(begx=begx     ,begy=begy,begz=begz,daz=daz,dz=dz,azim=azim+180.0,theta_die=12.0,theta_shift=4.0,dist_die=1.5,perp_die=1.0,thresh=200)
        self.fault(begx=begx+0.16,begy=begy,begz=begz,daz=daz,dz=dz,azim=azim      ,theta_die=12.0,theta_shift=4.0,dist_die=1.5,perp_die=1.0,thresh=200)
      else:
        self.fault(begx=begx,begy=begy     ,begz=begz,daz=daz,dz=dz,azim=azim+180.0,theta_die=12.0,theta_shift=4.0,dist_die=1.5,perp_die=1.0,thresh=200)
        self.fault(begx=begx,begy=begy+0.16,begz=begz,daz=daz,dz=dz,azim=azim      ,theta_die=12.0,theta_shift=4.0,dist_die=1.5,perp_die=1.0,thresh=200)
      # Move along x or y
      begx += dx; begy += dy

  def tinyfault(self,azim=0.0,begz=0.2,begx=0.5,begy=0.5,tscale=1.0,rand=True):
    """
    Puts in a tiny fault
    For now, will only give nice faults along 0,90,180,270 azimuths

    Parameters:
      azim - azimuth along which faults are oriented [0.0]
      begz - beginning position in z for fault [0.2]
      begx - beginning position in x for fault [0.5]
      begy - beginning position in x for fault [0.5]
      tscale - divides the shift in z by this amount
      rand - small random variations in the throw of faults [True]
    """
    daz=3000.0; dz = 3000.0
    if(rand):
      daz += np.random.rand()*(2*1000) - 1000.0
      dz  += np.random.rand()*(2*500)  - 500.0
    self.fault(begx=begx,begy=begy,begz=begz,daz=daz,dz=dz,azim=azim,theta_die=9.0,theta_shift=4.0,dist_die=0.3,perp_die=1.0)

  def smallfault(self,azim=0.0,begz=0.3,begx=0.5,begy=0.5,tscale=1.0,rand=True):
    """
    Puts in a small fault
    For now, will only give nice faults along 0,90,180,270 azimuths

    Parameters:
      azim   - azimuth along which faults are oriented [0.0]
      begz   - beginning position in z for fault [0.3]
      begx   - beginning position in x for fault [0.5]
      begy   - beginning position in x for fault [0.5]
      tscale - divides the shift in z by this amount
      rand   - small random variations in the throw of faults [True]
    """
    daz = 8000; dz = 5000
    if(rand):
      daz    += np.random.rand()*(2000) - 1000
      dz     += np.random.rand()*(2000) - 1000
      tscale += np.random.rand()*2
    self.fault(begx=begx,begy=begy,begz=begz,daz=daz,dz=dz,azim=azim,
        theta_die=11.0,theta_shift=4.0,dist_die=0.3,perp_die=1.0,throwsc=tscale,thresh=50/tscale)

  def mediumfault(self,azim=0.0,begz=0.6,begx=0.5,begy=0.5,tscale=1.0,rand=True):
    """
    Puts in a medium fault
    For now, will only give nice faults along 0,90,180,270 azimuths

    Parameters:
      azim   - azimuth along which faults are oriented [0.0]
      begz   - beginning position in z for fault [0.6]
      begx   - beginning position in x for fault [0.5]
      begy   - beginning position in x for fault [0.5]
      tscale - divides the shift z by this amount [10.0]
      rand   - small random variations in the throw of faults [True]
    """
    daz = 15000; dz = 12000
    if(rand):
      daz    += np.random.rand()*(2000) - 1000
      dz     += np.random.rand()*(2000) - 1000
      tscale += np.random.rand()*(2)
    self.fault(begx=begx,begy=begy,begz=begz,daz=daz,dz=dz,azim=azim,
        theta_die=11.0,theta_shift=4.0,dist_die=1.5,perp_die=1.0,throwsc=tscale,thresh=50/tscale)

  def largefault(self,azim=0.0,begz=0.6,begx=0.5,begy=0.5,tscale=6.0,rand=True):
    """
    Puts in a large fault
    For now, will only give nice faults along 0,90,180,270 azimuths

    Parameters:
      azim   - azimuth along which faults are oriented [0.0]
      begz   - beginning position in z for fault [0.6]
      begx   - beginning position in x for fault [0.5]
      begy   - beginning position in x for fault [0.5]
      tscale - divides the shift z by this amount [10.0]
      rand   - small random variations in the throw of faults [True]
    """
    daz = 25000; dz = 10000
    if(rand):
      daz    += np.random.rand()*(2000) - 1000
      dz     += np.random.rand()*(2000) - 1000
      tscale += np.random.rand()*(3)
    self.fault(begx=begx,begy=begy,begz=begz,daz=daz,dz=dz,azim=azim,
        theta_die=12.0,theta_shift=4.0,dist_die=1.5,perp_die=1.0,throwsc=tscale,thresh=200/tscale)

  def slidingfault(self,azim=0.0,begz=0.6,begx=0.5,begy=0.5,rand=True):
    """
    Puts in a sliding fault
    For now, will only give nice faults along 0,90,180,270

    Parameters:
      azim - azimuth along which faults are oriented [0.0]
      begz - beginning position in z for fault [0.6]
      begx - beginning position in x for fault [0.5]
      begy - beginning position in x for fault [0.5]
      rand - small random variations in the throw of faults [True]
    """
    daz = 10000; dz = 25000
    if(rand):
      daz += np.random.rand()*(2000) - 1000
      dz  += np.random.rand()*(2000) - 1000
    self.fault(begx=begx,begy=begy,begz=begz,daz=daz,dz=dz,azim=azim,theta_die=12.0,theta_shift=4.0,dist_die=1.5,perp_die=1.0,thresh=200)

  def squish(self,amp=100,azim=90.0,lam=0.1,rinline=0,rxline=0,npts=3,octaves=3,persist=0.6,mode='perlin'):
    """
    Folds the current geologic model along a specific azimuth.

    Parameters:
      amp     - The maximum amplitude of the folded event [100]
      azim    - The azimuth along which the event should be folded [90]
      lam     - The wavelength (lambda) of the fold
      rinline - Amount of random variation in the inline (fast spatial axis) direction
      rxline  - Amount of random variation in the crossline (sloww spatial axis) direction
    """
    nzin = self.vel.shape[2]
    # Allocate shift array
    nn = 3*max(self.__nx,self.__ny)
    shf = np.zeros([nn,nn],dtype='float32')
    if(mode == 'cos'):
      # Compute the maximum shift
      maxshift = int(amp/self.__dz)
      nzot = nzin + 2*maxshift
      # Expand the model
      velot = np.zeros([self.__ny,self.__nx,nzot],dtype='float32')
      lyrot = np.zeros([self.__ny,self.__nx,nzot],dtype='int32')
      self.ec8.expand(maxshift,maxshift,nzin,self.lyr,self.vel,nzot,lyrot,velot)
      # Fold the deposits on the expanded model
      self.ec8.squish(nzin,self.lyr,self.vel,shf,0,
                      azim,amp,lam,rinline,rxline,nzot,lyrot,velot)
    elif(mode == 'perlin'):
      # Compute the perlin shift function
      shf1d = noise_generator.perlin(x=np.linspace(0,npts,nn), octaves=octaves, period=80, Ngrad=80, persist=persist, ncpu=1)
      shf1d -= np.mean(shf1d); shf1d *= 10*amp
      shf = np.ascontiguousarray(np.tile(shf1d,(nn,1)).T).astype('float32')
      # Find the maximum shift to be applied
      pamp = np.max(np.abs(shf1d))
      maxshift = int(pamp/self.__dz)
      # Expand the model
      nzot = nzin + 2*maxshift
      velot = np.zeros([self.__ny,self.__nx,nzot],dtype='float32')
      lyrot = np.zeros([self.__ny,self.__nx,nzot],dtype='int32')
      self.ec8.expand(maxshift,maxshift,nzin,self.lyr,self.vel,nzot,lyrot,velot)
      # Fold the deposits on the expanded model
      self.ec8.squish(nzin,self.lyr,self.vel,shf,1,
                      azim,pamp,lam,rinline,rxline,nzot,lyrot,velot)
    # Update the model
    self.lyr = lyrot
    self.vel = velot

  def findsqlyrs(self,nlyrs,ntot,mindist):
    """
    Finds layer indices to squish. Makes sure that they layers
    are dist indices apart and are not the same

    Parameters:
      nlyrs - number of layers to squish
      ntot  - total number of layers to be deposited
      mindist - minimum distance between layers
    """
    # Get the first layer
    sqlyrs = []
    sqlyrs.append(np.random.randint(0,ntot))
    # Loop until all layers are found
    while(len(sqlyrs) < nlyrs):
      lidx = np.random.randint(0,ntot)
      # Compute distances
      nsq = len(sqlyrs)
      sqdist = np.zeros(nsq,dtype='int32')
      for isq in range(nsq):
        sqdist[isq] = np.abs(lidx - sqlyrs[isq])
      if(np.all(sqdist >= mindist)):
        sqlyrs.append(lidx)

    return sqlyrs

  def trim(self,top=0,bot=1000):
    """
    Trims the model in depth.
    This is useful for faulting large models and only portion of
    the model is desired and needs to be faulted

    Parameters:
      top: the top sample number at which to begin trimming
      bot: the bottom sample number at which to end trimming
    """
    nz = self.vel.shape[2]
    assert(bot > top),"Bottom sample %d must be larger than top %d"%(bot,top)
    assert(bot - top < nz), "Cannot trim more samples %d the current model size %d"%(bot-top,nz)
    # Only trim the label if it has been created
    if(self.lbl.shape[2] == nz):
      self.lbl = self.lbl[:,:,top:bot]
    self.vel = self.vel[:,:,top:bot]
    self.lyr = self.lyr[:,:,top:bot]

  def get_refl(self):
    """ Computes the reflectivity for the current velocity model """
    nz = self.vel.shape[2]
    ref = np.zeros(self.vel.shape,dtype='float32')
    velsm = gaussian_filter(self.vel,sigma=0.5).astype('float32')
    self.ec8.calcref(nz,velsm,ref)
    return ref

