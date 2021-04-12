"""
Useful functions for creating movies for presentations
and viewing frames of a numpy array
@author: Joseph Jennings
@version: 2020.02.25
"""
import os
import inpout.seppy as seppy
from inpout.seppy import bytes2float
from genutils.ptyprint import create_inttag, progressbar
import numpy as np
import matplotlib.pyplot as plt

def makemovie_mpl(arr,odir,ftype='png',qc=False,skip=1,pttag=False,**kwargs):
  """
  Saves each frame on the fast axis to a png for viewing

  Parameters:
    arr       - input array where the frames are on the fast axis
    odir      - the output directory where to save the frames
    ftype     - the type (extension) of figure to save [png]
    qc        - look at each frame as it is saved [False]
    skip      - whether to skip frames [1]
    pttag     - write the frame number in pretty (unix-friendly) format
    wbox      - the first dimension of the figure size (figure width) [10]
    hbox      - the second dimension of the figure size (figure height) [10]
    xmin      - the minimum lateral point in your data [0.0]
    xmax      - the maximum lateral point in your data (typically (nx-1)*dx) [nx]
    zmin      - the minimum depth point in your data [0.0]
    zmax      - the maximum depth point in your data (typically (nz-1)*dz) [nz]
    vmin      - the minumum value desired to be plotted (min(data))
    vmax      - the maximum value desired to be plotted (max(data))
    pclip     - a scale to be applied to shrink or expand the dynamic range
    xlabel    - the label of the x-axis [None]
    ylabel    - the label of the y-axis [None]
    labelsize - the fontsize of the labels [18]
    ticksize  - the fontsize of the ticks [18]
    ttlstring - title to be printed. Can be printed of the form ttlstring%(ottl + dttl*(framenumber))
    ottl      - the first title value to be printed
    dttl      - the sampling of the title value to be printed
    aratio    - aspect ratio of the figure [1.0]
  """
  # Check if directory exists
  if(os.path.isdir(odir) == False):
    os.mkdir(odir)
  nfr = arr.shape[2]
  k = 0
  frames = np.arange(0,nfr,skip)
  vmin = np.min(arr); vmax = np.max(arr)
  for ifr in progressbar(frames, "frames"):
    fig = plt.figure(figsize=(kwargs.get("wbox",10),kwargs.get("hbox",10)))
    ax = fig.add_subplot(111)
    ax.imshow(arr[:,:,ifr],cmap=kwargs.get("cmap","gray"),
        extent=[kwargs.get("xmin",0),kwargs.get("xmax",arr.shape[1]),
          kwargs.get("zmax",arr.shape[0]),kwargs.get("zmin",0)],
        vmin=kwargs.get("vmin",vmin)*kwargs.get('pclip',1.0),
        vmax=kwargs.get("vmax",vmax)*kwargs.get('pclip',1.0))
    ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',18))
    ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',18))
    ax.tick_params(labelsize=kwargs.get('ticksize',18))
    if('%' in kwargs.get('ttlstring'),''):
      ax.set_title(kwargs.get('ttlstring','')%(kwargs.get('ottl',0.0) + kwargs.get('dttl',1.0)*k),
          fontsize=kwargs.get('labelsize',18))
    ax.set_aspect(kwargs.get('aratio',1.0))
    if(pttag):
      tag = create_inttag(k,nfr)
    else:
      tag = str(k)
    plt.savefig(odir+'/'+tag+'.'+ftype,bbox_inches='tight',dpi=kwargs.get("dpi",150))
    k += 1
    if(qc):
      plt.show()

def makemoviesbs_mpl(arr1,arr2,odir,ftype='png',qc=False,skip=1,pttag=False,**kwargs):
  """
  Saves each frame on the fast axis to a png for viewing

  Parameters:
    arr1  - first input array (left panel) where the frames are on the fast axis
    arr2  - second input array (right panel) where the frames are on the fast axis
    odir  - the output directory where to save the frames
    ftype - the type (extension) of figure to save [png]
    qc    - look at each frame as it is saved [False]
    skip  - whether to skip frames [1]
    pttag - write the frame number in pretty (unix-friendly) format
  """
  # Check if directory exists
  if(os.path.isdir(odir) == False):
    os.mkdir(odir)
  assert(arr1.shape[2] == arr2.shape[2]),"Both movies must have same number of frames"
  nfr = arr1.shape[2]
  k = 0
  frames = np.arange(0,nfr,skip)
  vmin1 = np.min(arr1); vmax1 = np.max(arr1)
  vmin2 = np.min(arr2); vmax2 = np.max(arr2)
  for ifr in progressbar(frames, "frames"):
    # Setup plot
    f,ax = plt.subplots(1,2,figsize=(kwargs.get("figsize1",15),kwargs.get("figsize2",8)),
        gridspec_kw={'width_ratios': [kwargs.get("wratio1",1), kwargs.get("wratio2",1)]})
    # First plot
    im1 = ax[0].imshow(arr1[:,:,ifr],cmap=kwargs.get("cmap1","gray"),
        extent=[kwargs.get("xmin1",0),kwargs.get("xmax1",arr1.shape[1]),
          kwargs.get("zmax1",arr1.shape[0]),kwargs.get("zmin1",0)],
        vmin=kwargs.get("vmin1",vmin1)*kwargs.get('pclip',1.0),
        vmax=kwargs.get("vmax1",vmax1)*kwargs.get('pclip',1.0),aspect=kwargs.get('aspect1',1.0))
    ax[0].set_xlabel(kwargs.get('xlabel1',''),fontsize=kwargs.get('labelsize',18))
    ax[0].set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',18))
    ax[0].tick_params(labelsize=kwargs.get('ticksize',18))
    if('%' in kwargs.get('ttlstring1'),''):
      ax[0].set_title(kwargs.get('ttlstring1','')%(kwargs.get('ottl1',0.0) + kwargs.get('dttl1',1.0)*k),
          fontsize=kwargs.get('labelsize',18))
    else:
      ax[0].set_title(kwargs.get('ttlstring1',''),fontsize=kwargs.get('labelsize',18))
    # Second plot
    im2 = ax[1].imshow(arr2[:,:,ifr],cmap=kwargs.get("cmap2","gray"),
        extent=[kwargs.get("xmin2",0),kwargs.get("xmax2",arr2.shape[1]),
          kwargs.get("zmax2",arr2.shape[0]),kwargs.get("zmin2",0)],
        vmin=kwargs.get("vmin2",vmin2)*kwargs.get('pclip',1.0),
        vmax=kwargs.get("vmax2",vmax2)*kwargs.get('pclip',1.0),aspect=kwargs.get('aspect2',1.0))
    ax[1].set_xlabel(kwargs.get('xlabel2',''),fontsize=kwargs.get('labelsize',18))
    ax[1].set_ylabel(kwargs.get('ylabel2',''),fontsize=kwargs.get('labelsize',18))
    ax[1].tick_params(labelsize=kwargs.get('ticksize',18))
    if('%' in kwargs.get('ttlstring2','')):
      ax[1].set_title(kwargs.get('ttlstring2','')%(kwargs.get('ottl2',0.0) + kwargs.get('dttl2',1.0)*k),
          fontsize=kwargs.get('labelsize',18))
    else:
      ax[1].set_title(kwargs.get('ttlstring2',''),fontsize=kwargs.get('labelsize',18))
    # Color bar
    if(kwargs.get('cbar',False)):
      cbar_ax = f.add_axes([kwargs.get('barx',0.91),kwargs.get('barz',0.12),
        kwargs.get('wbar',0.02),kwargs.get('hbar',0.75)])
      cbar = f.colorbar(im2,cbar_ax,format='%.2f')
      cbar.ax.tick_params(labelsize=kwargs.get('ticksize',18))
      cbar.set_label(kwargs.get('barlabel',''),fontsize=kwargs.get("barlabelsize",18))
      cbar.draw_all()
    # Spacing between plots
    plt.subplots_adjust(wspace=kwargs.get("pltspace",0.2))
    if(pttag):
      tag = create_inttag(k,nfr)
    else:
      tag = str(k)
    plt.savefig(odir+'/'+tag+'.'+ftype,bbox_inches='tight',dpi=kwargs.get("dpi",150))
    k += 1
    if(qc):
      plt.show()

def makemovietb_mpl(arr1,arr2,odir,ftype='png',qc=False,skip=1,pttag=False,**kwargs):
  """
  Saves each frame on the fast axis to a png for viewing

  Parameters:
    arr1  - first input array (Top panel) where the frames are on the fast axis
    arr2  - second input array (Bottom panel) where the frames are on the fast axis
    odir  - the output directory where to save the frames
    ftype - the type (extension) of figure to save [png]
    qc    - look at each frame as it is saved [False]
    skip  - whether to skip frames [1]
    pttag - write the frame number in pretty (unix-friendly) format
  """
  # Check if directory exists
  if(os.path.isdir(odir) == False):
    os.mkdir(odir)
  assert(arr1.shape[2] == arr2.shape[2]),"Both movies must have same number of frames"
  nfr = arr1.shape[2]
  k = 0
  frames = np.arange(0,nfr,skip)
  vmin1 = np.min(arr1); vmax1 = np.max(arr1)
  vmin2 = np.min(arr2); vmax2 = np.max(arr2)
  for ifr in progressbar(frames, "frames"):
    # Setup plot
    f,ax = plt.subplots(2,1,figsize=(kwargs.get("wbox",15),kwargs.get("hbox",8)))
    # First plot
    im1 = ax[0].imshow(arr1[:,:,ifr],cmap=kwargs.get("cmap1","gray"),
        extent=[kwargs.get("xmin1",0),kwargs.get("xmax1",arr1.shape[1]),
          kwargs.get("zmax1",arr1.shape[0]),kwargs.get("zmin1",0)],
        vmin=kwargs.get("vmin1",vmin1)*kwargs.get('pclip',1.0),
        vmax=kwargs.get("vmax1",vmax1)*kwargs.get('pclip',1.0),aspect=kwargs.get('aspect1',1.0))
    ax[0].set_ylabel(kwargs.get('ylabel1',''),fontsize=kwargs.get('labelsize',18))
    ax[0].set_xlabel(kwargs.get('xlabel1',''),fontsize=kwargs.get('labelsize',18))
    ax[0].tick_params(labelsize=kwargs.get('ticksize',18))
    if('%' in kwargs.get('ttlstring1'),''):
      ax[0].set_title(kwargs.get('ttlstring1','')%(kwargs.get('ottl1',0.0) + kwargs.get('dttl1',1.0)*k),
          fontsize=kwargs.get('labelsize',18))
    else:
      ax[0].set_title(kwargs.get('ttlstring1',''),fontsize=kwargs.get('labelsize',18))
    # Second plot
    im2 = ax[1].imshow(arr2[:,:,ifr],cmap=kwargs.get("cmap2","gray"),
        extent=[kwargs.get("xmin2",0),kwargs.get("xmax2",arr2.shape[1]),
          kwargs.get("zmax2",arr2.shape[0]),kwargs.get("zmin2",0)],
        vmin=kwargs.get("vmin2",vmin2)*kwargs.get('pclip',1.0),
        vmax=kwargs.get("vmax2",vmax2)*kwargs.get('pclip',1.0),aspect=kwargs.get('aspect2',1.0))
    ax[1].set_xlabel(kwargs.get('xlabel2',''),fontsize=kwargs.get('labelsize',18))
    ax[1].tick_params(labelsize=kwargs.get('ticksize',18))
    ax[1].set_ylabel(kwargs.get('ylabel2',''),fontsize=kwargs.get('labelsize',18))
    if('%' in kwargs.get('ttlstring2','')):
      ax[1].set_title(kwargs.get('ttlstring2','')%(kwargs.get('ottl2',0.0) + kwargs.get('dttl2',1.0)*k),
          fontsize=kwargs.get('labelsize',18))
    else:
      ax[1].set_title(kwargs.get('ttlstring2',''),fontsize=kwargs.get('labelsize',18))
    # Color bar
    if(kwargs.get('cbar',False)):
      cbar_ax = f.add_axes([kwargs.get('barx',0.91),kwargs.get('barz',0.12),
        kwargs.get('wbar',0.02),kwargs.get('hbar',0.75)])
      cbar = f.colorbar(im2,cbar_ax,format='%.2f')
      cbar.ax.tick_params(labelsize=kwargs.get('ticksize',18))
      cbar.set_label(kwargs.get('barlabel',''),fontsize=kwargs.get("barlabelsize",18))
      cbar.draw_all()
    # Spacing between plots
    plt.subplots_adjust(wspace=kwargs.get("pltspace",0.2))
    if(pttag):
      tag = create_inttag(k,nfr)
    else:
      tag = str(k)
    plt.savefig(odir+'/'+tag+'.'+ftype,bbox_inches='tight',dpi=kwargs.get("dpi",150))
    k += 1
    if(qc):
      plt.show()

#TODO: replace xmin,xmax, etc with ox,dx
def viewimgframeskey(data,transp=True,fast=True,show=True,**kwargs):
  """
  Provides a frame by frame interactive viewing of a 3D numpy array via the arrow keys.
  Assumes the slow axis is the first axis.

  Parameters:
    data      - an input 3D array. Must be 3D otherwise will fail
    transp    - whether to transpose the first and second axes before plotting
    cmap      - colormap [gray]
    vmin      - minimum value to display in the data [default is minimum amplitude of all data]
    vmax      - maximum value to display in the data [default is maximum amplitude of all data]
    pclip     - how much to clip the min and max of the amplitudes [0.9]
    ttlstring - title to be printed. Can be printed of the form ttlstring%(ottl + dttl*(framenumber))
    ottl      - origin for printing title values [0.0]
    dttl      - sampling for printing title values [1.0]
    scalebar  - flag of whether to plot a colorbar (False)
    hbar      - height of the scale bar
    wbar      - width of the scale bar
    zbar      - z position of scale bar
    xbar      - x position of scale bar
    barlabel  - colorbar label
    interp    - interpolation type for better display of the data (sinc for seismic, bilinear of velocity) [none]
    show      - flag for calling plt.show() [True]
  """
  if(type(data) is list):
    data = np.concatenate([iimg[np.newaxis] for iimg in data],axis=0)
  if(len(data.shape) < 3):
    raise Exception("Data must be 3D")
  [nex,nx,nz] = data.shape
  xmin = kwargs.get('ox',0.0)
  xmax = kwargs.get('ox',0.0) + nx*kwargs.get('dx',1.0)
  zmin = kwargs.get('oz',0.0)
  zmax = kwargs.get('oz',0.0) + nz*kwargs.get('dz',1.0)
  curr_pos = 0
  vmin = kwargs.get('vmin',None); vmax = kwargs.get('vmax',None)
  if(vmin == None or vmax == None):
    vmin = np.min(data)*kwargs.get('pclip',0.9)
    vmax = np.max(data)*kwargs.get('pclip',0.9)

  def key_event(e):
    nonlocal curr_pos,vmin,vmax

    if e.key == "n":
        curr_pos = curr_pos + 1
    elif e.key == "m":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % data.shape[0]

    if(transp):
      img = data[curr_pos,:,:].T
    else:
      img = data[curr_pos,:,:]
    if(not fast):
      ax.cla()
      ax.imshow(img,cmap=kwargs.get('cmap','gray'),vmin=vmin,vmax=vmax,
          extent=[xmin,xmax,zmax,zmin],
          interpolation=kwargs.get('interp','none'),aspect='auto')
    ax.set_title('%d'%(curr_pos),fontsize=kwargs.get('labelsize',14))
    ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',14))
    ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',14))
    #if(kwargs.get('scalebar',False)):
    #  cbar_ax = fig.add_axes([kwargs.get('barx',0.91),kwargs.get('barz',0.12),
    #    kwargs.get('wbar',0.02),kwargs.get('hbar',0.75)])
    #  cbar = fig.colorbar(l,cbar_ax,format='%.2f')
    #  cbar.ax.tick_params(labelsize=kwargs.get('ticksize',18))
    #  cbar.set_label(kwargs.get('barlabel',''),fontsize=kwargs.get("barlabelsize",18))
    #  cbar.draw_all()
    if('%' in kwargs.get('ttlstring',' ')):
      ax.set_title(kwargs.get('ttlstring',' ')%(kwargs.get('ottl',0.0) + kwargs.get('dttl',1.0)*curr_pos),
          fontsize=kwargs.get('labelsize',14))
    else:
      ax.set_title(kwargs.get('ttlstring','%d'%curr_pos),fontsize=kwargs.get('labelsize',14))
    ax.tick_params(labelsize=kwargs.get('ticksize',14))
    if(fast):
      l.set_data(img)
    fig.canvas.draw()

  fig = plt.figure(figsize=(kwargs.get("wbox",10),kwargs.get("hbox",10)))
  fig.canvas.mpl_connect('key_press_event', key_event)
  ax = fig.add_subplot(111)
  # Show the first frame
  if(transp):
    img = data[0,:,:].T
  else:
    img = data[0,:,:]
  l = ax.imshow(img,cmap=kwargs.get('cmap','gray'),vmin=vmin,vmax=vmax,
      extent=[xmin,xmax,zmax,zmin],
      interpolation=kwargs.get('interp','none'),aspect='auto')
  ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',14))
  ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',14))
  ax.tick_params(labelsize=kwargs.get('ticksize',14))
  # Color bar
  if(kwargs.get('scalebar',False)):
    cbar_ax = fig.add_axes([kwargs.get('barx',0.91),kwargs.get('barz',0.12),
      kwargs.get('wbar',0.02),kwargs.get('hbar',0.75)])
    cbar = fig.colorbar(l,cbar_ax,format='%.2f')
    cbar.ax.tick_params(labelsize=kwargs.get('ticksize',18))
    cbar.set_label(kwargs.get('barlabel',''),fontsize=kwargs.get("barlabelsize",18))
    cbar.draw_all()
  if('%' in kwargs.get('ttlstring',' ')):
    ax.set_title(kwargs.get('ttlstring',' ')%(kwargs.get('ottl',0.0)),fontsize=kwargs.get('labelsize',14))
  else:
    ax.set_title(kwargs.get('ttlstring','%d'%curr_pos),fontsize=kwargs.get('labelsize',14))
  ax.tick_params(labelsize=kwargs.get('ticksize',14))
  if(show):
    plt.show()

def viewpltframeskey(data,ox=0.0,dx=1.0,transp=True,show=True,**kwargs):
  """
  Provides a frame by frame interactive viewing of a 2D numpy array via the arrow keys.
  Assumes the slow axis is the first axis.

  Parameters:
    data      - an input 2D array. Must be 2D otherwise will fail
    vmin      - minimum value to display in the data [default is minimum amplitude of all data]
    vmax      - maximum value to display in the data [default is maximum amplitude of all data]
    pclip     - how much to clip the min and max of the amplitudes [0.9]
    ttlstring - title to be printed. Can be printed of the form ttlstring%(ottl + dttl*(framenumber))
    ottl      - origin for printing title values [0.0]
    dttl      - sampling for printing title values [1.0]
    show      - flag for calling plt.show() [True]
  """
  if(len(data.shape) < 2):
    raise Exception("Data must be 2D")
  curr_pos = 0
  # Create the x axis
  nx = data.shape[1]
  xs = np.linspace(ox,ox+(nx-1)*dx,nx)
  # Find the min and the max of the frames
  if(kwargs.get('ymin',None) == None or kwargs.get('ymax',None) == None):
    ymin = np.min(data); ymax = np.max(data)

  def key_event(e):
    nonlocal curr_pos,xs

    if e.key == "n":
        curr_pos = curr_pos + 1
    elif e.key == "m":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % data.shape[0]

    if(transp):
      crv = data[curr_pos,:].T
    else:
      crv = data[curr_pos,:]
    ax.cla()
    ax.plot(xs,crv,color=kwargs.get('color','tab:blue'),linewidth=kwargs.get('linewidth',1.5))
    ax.set_ylim([kwargs.get('ymin',ymin),kwargs.get('ymax',ymax)])
    ax.set_title('%d'%(curr_pos),fontsize=kwargs.get('labelsize',14))
    ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',14))
    ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',14))
    if('%' in kwargs.get('ttlstring',' ')):
      ax.set_title(kwargs.get('ttlstring',' ')%(kwargs.get('ottl',0.0) + kwargs.get('dttl',1.0)*curr_pos),
          fontsize=kwargs.get('labelsize',14))
    else:
      ax.set_title(kwargs.get('ttlstring','%d'%curr_pos),fontsize=kwargs.get('labelsize',14))
    ax.tick_params(labelsize=kwargs.get('ticksize',14))
    fig.canvas.draw()

  fig = plt.figure(figsize=(kwargs.get("wbox",10),kwargs.get("hbox",10)))
  fig.canvas.mpl_connect('key_press_event', key_event)
  ax = fig.add_subplot(111)
  # Show the first frame
  if(transp):
    crv = data[0,:].T
  else:
    crv = data[0,:,:]
  l = ax.plot(xs,crv,color=kwargs.get('color','tab:blue'),linewidth=kwargs.get('linewidth',1.5))
  ax.set_ylim([kwargs.get('ymin',ymin),kwargs.get('ymax',ymax)])
  ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',14))
  ax.set_ylabel(kwargs.get('ylabel',''),fontsize=kwargs.get('labelsize',14))
  ax.tick_params(labelsize=kwargs.get('ticksize',14))
  if('%' in kwargs.get('ttlstring',' ')):
    ax.set_title(kwargs.get('ttlstring',' ')%(kwargs.get('ottl',0.0)),fontsize=kwargs.get('labelsize',14))
  else:
    ax.set_title(kwargs.get('ttlstring','%d'%curr_pos),fontsize=kwargs.get('labelsize',14))
  ax.tick_params(labelsize=kwargs.get('ticksize',14))
  if(show):
    plt.show()

def viewframessld(data,transp=True,**kwargs):
  pass

def viewcube3d(data,os=[0.0,0.0,0.0],ds=[1.0,1.0,1.0],show=True,**kwargs):
  """
  Plots three frames of a 3D plot for visualization. Allows
  for user interaction with the arrow keys or e,w,n,s keys.
  Originally written by Huy Le with some of my modifications

  Parameters:
    data - input data cube (numpy array)
    os   - origins of each axis [0.0,0.0,0.0]
    ds   - samplings of each axis [1.0,1.0,1.0]
  """
  # Transpose if requested
  if(not kwargs.get('transp',False)):
    data = np.expand_dims(data,axis=0)
    data = np.transpose(data,(0,1,3,2))
  else:
    data = (np.expand_dims(data,axis=-1)).T
    data = np.transpose(data,(0,1,3,2))
  # Get the shape of the cube
  ns = np.flip(data.shape)
  # Get the datatype
  byte = True if data.dtype == 'uint8' else False
  # Make the coordinates for the cross hairs
  ds = np.append(np.flip(ds),1.0)
  os = np.append(np.flip(os),0.0)
  x1=np.linspace(os[0], os[0] + ds[0]*(ns[0]), ns[0])
  x2=np.linspace(os[1], os[1] + ds[1]*(ns[1]), ns[1])
  x3=np.linspace(os[2], os[2] + ds[2]*(ns[2]), ns[2])

  # Compute plotting min and max
  vmin = kwargs.get('vmin',None); vmax = kwargs.get('vmax',None)
  if(vmin == None or vmax == None):
    vmin = np.min(data)*kwargs.get('pclip',1.0)
    vmax = np.max(data)*kwargs.get('pclip',1.0)
    if(byte):
      vmin -= 255/2.; vmin *= 1./255
      vmax -= 255/2.; vmax *= 1./255

  # Define nonlocal variables
  loc1 = kwargs.get('loc1',ns[0]/2*ds[0]+os[0])
  i1 = int((loc1 - os[0])/ds[0])
  loc2 = kwargs.get('loc2',ns[1]/2*ds[1]+os[1])
  i2 = int((loc2 - os[1])/ds[1])
  loc3 = kwargs.get('loc3',ns[2]/2*ds[2]+os[2])
  i3 = int((loc3 - os[2])/ds[2])
  ax1 = None; ax2 = None; ax3 = None; ax4 = None
  curr_pos = 0; inaxes = None; updated = False
  # Govern the keyboard movement
  j1 = kwargs.get('j1',1); j2 = kwargs.get('j2',1); j3 = kwargs.get('j3',1)

  # Axis labels
  label1 = kwargs.get('label1',' '); label2 = kwargs.get('label2',' '); label3 = kwargs.get('label3', ' ')

  def key_event(e):
    nonlocal i1,loc1,i2,loc2,i3,loc3,ax1,ax2,ax3,ax4,curr_pos

    if(ax[1,0].get_xlim()[0] == os[0] and ax[1,0].get_ylim()[1] == os[2] and ax[1,1].get_xlim()[0] == os[1]):
      zoomed_out = True
    else:
      zoomed_out = False

    if e.key=="u" or e.key=="d":
      if e.key=="u":
        i3-=j3
      elif e.key=="d":
        i3+=j3
      i3=i3%ns[2]
      loc3=i3*ds[2]+os[2]

      if(zoomed_out):
        ax[0,0].cla()
      else:
        del ax[0,0].lines[:]

      slc = bytes2float(data[curr_pos,i3,:,:]) if byte else data[curr_pos,i3,:,:]
      ax[0,0].imshow(np.flip(slc,0),interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[0],os[0]+(ns[0])*ds[0],os[1],os[1]+(ns[1])*ds[1]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[0,0].set_ylabel(label2,fontsize=kwargs.get('labelsize',14))
      ax[0,0].tick_params(labelsize=kwargs.get('ticksize',14))
      del ax[0,0].lines[:]
      ax[0,0].plot(loc1*np.ones((ns[1],)),x2,c='k')
      ax[0,0].plot(x1,loc2*np.ones((ns[0],)),c='k')

      del ax[1,0].lines[:]
      ax[1,0].plot(loc1*np.ones((ns[2],)),x3,c='k')
      ax[1,0].plot(x1,loc3*np.ones((ns[0],)),c='k')

      del ax[1,1].lines[:]
      ax[1,1].plot(loc2*np.ones((ns[2],)),x3,c='k')
      ax[1,1].plot(x2,loc3*np.ones((ns[1],)),c='k')
      ax1.set_yticks([loc3])
      ax1.set_yticklabels(['%.2f'%(loc3)],rotation='vertical',va='center')
      ax1.tick_params(labelsize=kwargs.get('ticksize',14))
      ax2.set_xticks([loc2])
      ax2.set_xticklabels(['%.2f'%(loc2)])
      ax2.tick_params(labelsize=kwargs.get('ticksize',14))

    elif(e.key=="e" or e.key=="w"):
      if(e.key=="w"):
        i1-=j1
      elif(e.key=="e"):
        i1+=j1
      i1=i1%ns[0]
      loc1=i1*ds[0]+os[0]

      if(zoomed_out):
        ax[1,1].cla()
      else:
        del ax[1,1].lines[:]

      slc = bytes2float(data[curr_pos,:,:,i1]) if byte else data[curr_pos,:,:,i1]
      ax[1,1].imshow(slc,interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[1],os[1]+(ns[1])*ds[1],os[2]+(ns[2])*ds[2],os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[1,1].set_xlabel(label2,fontsize=kwargs.get('labelsize',14))
      ax[1,1].tick_params(labelsize=kwargs.get('ticksize',14))
      del ax[1,1].lines[:]
      ax[1,1].plot(loc2*np.ones((ns[2],)),x3,c='k')
      ax[1,1].plot(x2,loc3*np.ones((ns[1],)),c='k')
      ax1.set_yticks([loc3])
      ax1.set_yticklabels(['%.2f'%(loc3)],rotation='vertical',va='center')
      ax1.tick_params(labelsize=kwargs.get('ticksize',14))
      ax2.set_xticks([loc2])
      ax2.set_xticklabels(['%.2f'%(loc2)])
      ax2.tick_params(labelsize=kwargs.get('ticksize',14))

      del ax[1,0].lines[:]
      ax[1,0].plot(loc1*np.ones((ns[2],)),x3,c='k')
      ax[1,0].plot(x1,loc3*np.ones((ns[0],)),c='k')

      del ax[0,0].lines[:]
      ax[0,0].plot(loc1*np.ones((ns[1],)),x2,c='k')
      ax[0,0].plot(x1,loc2*np.ones((ns[0],)),c='k')
      ax3.set_yticks([loc2])
      ax3.set_yticklabels(['%.2f'%(loc2)],rotation='vertical',va='center')
      ax3.tick_params(labelsize=kwargs.get('ticksize',14))
      ax4.set_xticks([loc1])
      ax4.set_xticklabels(['%.2f'%(loc1)])
      ax4.tick_params(labelsize=kwargs.get('ticksize',14))

    elif e.key=="m" or e.key=="n":
      if e.key=="m":
        i2-=j2
      elif e.key=="n":
        i2+=j2
      i2=i2%ns[1]
      loc2=i2*ds[1]+os[1]


      if(zoomed_out):
        ax[1,0].cla()
      else:
        del ax[1,0].lines[:]

      slc = bytes2float(data[curr_pos,:,i2,:]) if byte else data[curr_pos,:,i2,:]
      ax[1,0].imshow(slc,interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[0],os[0]+(ns[0])*ds[0],os[2]+ds[2]*(ns[2]),os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[1,0].tick_params(labelsize=kwargs.get('ticksize',14))
      del ax[1,1].lines[:]
      ax[1,0].plot(loc1*np.ones((ns[2],)),x3,c='k')
      ax[1,0].plot(x1,loc3*np.ones((ns[0],)),c='k')
      ax[1,0].set_xlabel(label1,fontsize=kwargs.get('labelsize',14))
      ax[1,0].set_ylabel(label3,fontsize=kwargs.get('labelsize',14))

      del ax[1,1].lines[:]
      ax[1,1].plot(loc2*np.ones((ns[2],)),x3,c='k')
      ax[1,1].plot(x2,loc3*np.ones((ns[1],)),c='k')
      ax1.set_yticks([loc3])
      ax1.set_yticklabels(['%.2f'%(loc3)],rotation='vertical',va='center')
      ax1.tick_params(labelsize=kwargs.get('ticksize',14))
      ax2.set_xticks([loc2])
      ax2.set_xticklabels(['%.2f'%(loc2)])
      ax2.tick_params(labelsize=kwargs.get('ticksize',14))

      del ax[0,0].lines[:]
      ax[0,0].plot(loc1*np.ones((ns[1],)),x2,c='k')
      ax[0,0].plot(x1,loc2*np.ones((ns[0],)),c='k')
      ax3.set_yticks([loc2])
      ax3.set_yticklabels(['%.2f'%(loc2)],rotation='vertical',va='center')
      ax3.tick_params(labelsize=kwargs.get('ticksize',14))
      ax4.set_xticks([loc1])
      ax4.set_xticklabels(['%.2f'%(loc1)])
      ax4.tick_params(labelsize=kwargs.get('ticksize',14))

    elif e.key=="left" or e.key=="right" or e.key.isdigit():
      if e.key=="left":
        curr_pos-=1
      if e.key=="right":
        curr_pos+=1
      if e.key.isdigit():
        curr_pos=int(e.key)
      curr_pos=curr_pos%ns[3]

      if(zoomed_out):
        ax[0,1].cla()
      else:
        del ax[0,1].lines[:]

      ax[0,1].get_xaxis().set_visible(False)
      ax[0,1].get_yaxis().set_visible(False)
      ax[0,1].axis('off')
      ax[0,1].text(0.5,0.5,title[curr_pos],horizontalalignment='center',verticalalignment='center',fontsize=50)

      if(zoomed_out):
        ax[1,0].cla()
      else:
        del ax[1,0].lines[:]
      slc = bytes2float(data[curr_pos,:,i2,:]) if byte else data[curr_pos,:,i2,:]
      ax[1,0].imshow(slc,interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[0],os[0]+(ns[0])*ds[0],os[2]+ds[2]*(ns[2]),os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[1,0].plot(loc1*np.ones((ns[2],)),x3,c='k')
      ax[1,0].plot(x1,loc3*np.ones((ns[0],)),c='k')
      ax[1,0].set_xlabel(label1,fontsize=kwargs.get('labelsize',14))
      ax[1,0].set_ylabel(label3,fontsize=kwargs.get('labelsize',14))

      # yz plane
      if(zoomed_out):
        ax[1,1].cla()
      else:
        del ax[1,1].lines[:]
      slc = bytes2float(data[curr_pos,:,:,i1]) if byte else data[curr_pos,:,:,i1]
      ax[1,1].imshow(slc,interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[1],os[1]+(ns[1])*ds[1],os[2]+(ns[2])*ds[2],os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[1,1].plot(loc2*np.ones((ns[2],)),x3,c='k')
      ax[1,1].plot(x2,loc3*np.ones((ns[1],)),c='k')
      ax[1,1].set_xlabel(label2,fontsize=kwargs.get('labelsize',14))
      ax1.set_yticks([loc3])
      ax1.set_yticklabels(['%.2f'%(loc3)],rotation='vertical',va='center')
      ax1.tick_params(labelsize=kwargs.get('ticksize',14))
      ax2.set_xticks([loc2])
      ax2.set_xticklabels(['%.2f'%(loc2)])
      ax2.tick_params(labelsize=kwargs.get('ticksize',14))

      # xy plane
      if(zoomed_out):
        ax[0,0].cla()
      else:
        del ax[0,0].lines[:]
      slc = bytes2float(data[curr_pos,i3,:,:]) if byte else data[curr_pos,i3,:,:]
      ax[0,0].imshow(np.flip(slc,0),interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[0],os[0]+(ns[0])*ds[0],os[1],os[1]+(ns[1])*ds[1]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[0,0].plot(loc1*np.ones((ns[1],)),x2,c='k')
      ax[0,0].plot(x1,loc2*np.ones((ns[0],)),c='k')
      ax[0,0].set_ylabel(label2,fontsize=kwargs.get('labelsize',14))
      ax3.set_yticks([loc2])
      ax3.set_yticklabels(['%.2f'%(loc2)],rotation='vertical',va='center')
      ax3.tick_params(labelsize=kwargs.get('ticksize',14))
      ax4.set_xticks([loc1])
      ax4.set_xticklabels(['%.2f'%(loc1)])
      ax4.tick_params(labelsize=kwargs.get('ticksize',14))

    ax[0,0].set_xlim(ax[1,0].get_xlim())
    ax[1,1].set_ylim(ax[1,0].get_ylim())
    fig.canvas.draw()

  def onclick(e):
    nonlocal i1,loc1,i2,loc2,i3,loc3,ax1,ax2,ax3,ax4,curr_pos,inaxes,updated
    tb = plt.get_current_fig_manager().toolbar
    if(tb.mode == ""):
      if e.inaxes==ax1 or e.inaxes==ax2:
        loc2=e.xdata
        i2=int((loc2-os[1])/ds[1])
        loc2=i2*ds[1]+os[1]
        loc3=e.ydata
        i3=int((loc3-os[2])/ds[2])
        loc3=i3*ds[2]+os[2]
      if e.inaxes==ax3 or e.inaxes==ax4:
        loc1=e.xdata
        i1=int((loc1-os[0])/ds[0])
        loc1=i1*ds[0]+os[0]
        loc2=e.ydata
        i2=int((loc2-os[1])/ds[1])
        loc2=i2*ds[1]+os[1]
      if e.inaxes==ax[1,0]:
        loc1=e.xdata
        i1=int((loc1-os[0])/ds[0])
        loc1=i1*ds[0]+os[0]
        loc3=e.ydata
        i3=int((loc3-os[2])/ds[2])
        loc3=i3*ds[2]+os[2]

      if(ax[1,0].get_xlim()[0] == os[0] and ax[1,0].get_ylim()[1] == os[2] and ax[1,1].get_xlim()[0] == os[1]):
        zoomed_out = True
      else:
        zoomed_out = False

      if(zoomed_out):
        ax[1,0].cla()
      else:
        del ax[1,0].lines[:]
      slc = bytes2float(data[curr_pos,:,i2,:]) if byte else data[curr_pos,:,i2,:]
      ax[1,0].imshow(slc,interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[0],os[0]+(ns[0])*ds[0],os[2]+ds[2]*(ns[2]),os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[1,0].plot(loc1*np.ones((ns[2],)),x3,c='k')
      ax[1,0].plot(x1,loc3*np.ones((ns[0],)),c='k')
      ax[1,0].set_xlabel(label1,fontsize=kwargs.get('labelsize',14))
      ax[1,0].set_ylabel(label3,fontsize=kwargs.get('labelsize',14))

      ## yz plane
      if(zoomed_out):
        ax[1,1].cla()
      else:
        del ax[1,1].lines[:]
      slc = bytes2float(data[curr_pos,:,:,i1]) if byte else data[curr_pos,:,:,i1]
      ax[1,1].imshow(slc,interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[1],os[1]+(ns[1])*ds[1],os[2]+(ns[2])*ds[2],os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[1,1].plot(loc2*np.ones((ns[2],)),x3,c='k')
      ax[1,1].plot(x2,loc3*np.ones((ns[1],)),c='k')
      ax[1,1].set_xlabel(label2,fontsize=kwargs.get('labelsize',14))
      ax1.set_yticks([loc3])
      ax1.set_yticklabels(['%.2f'%(loc3)],rotation='vertical',va='center')
      ax1.tick_params(labelsize=kwargs.get('ticksize',14))
      ax2.set_xticks([loc2])
      ax2.set_xticklabels(['%.2f'%(loc2)])
      ax2.tick_params(labelsize=kwargs.get('ticksize',14))

      ## xy plane
      if(zoomed_out):
        ax[0,0].cla()
      else:
        del ax[0,0].lines[:]
      slc = bytes2float(data[curr_pos,i3,:,:]) if byte else data[curr_pos,i3,:,:]
      ax[0,0].imshow(np.flip(slc,0),interpolation=kwargs.get('interp','none'),aspect='auto',
          extent=[os[0],os[0]+(ns[0])*ds[0],os[1],os[1]+(ns[1])*ds[1]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
      ax[0,0].plot(loc1*np.ones((ns[1],)),x2,c='k')
      ax[0,0].plot(x1,loc2*np.ones((ns[0],)),c='k')
      ax[0,0].set_ylabel(label2,fontsize=kwargs.get('labelsize',14))
      ax3.set_yticks([loc2])
      ax3.set_yticklabels(['%.2f'%(loc2)],rotation='vertical',va='center')
      ax4.set_xticks([loc1])
      ax4.set_xticklabels(['%.2f'%(loc1)])

      fig.canvas.draw()
    else:
      # Handle the zoom
      inaxes = e.inaxes
      updated = False

  def ondraw(e):
    nonlocal ax3,ax4,inaxes,updated
    if(plt.get_current_fig_manager().toolbar.mode == 'zoom rect' and updated == False):
      if (inaxes==ax1 or inaxes==ax2):
        ax[1,0].set_ylim(ax[1,1].get_ylim())
        ax[0,0].set_ylim(ax[1,1].get_xlim())
        updated = True
      if(inaxes==ax3 or inaxes==ax4):
        ax[1,0].set_xlim(ax[0,0].get_xlim())
        ax[1,1].set_xlim(ax[0,0].get_ylim())
        updated = True
      if(inaxes == ax[1,0]):
        ax[0,0].set_xlim(ax[1,0].get_xlim())
        ax[1,1].set_ylim(ax[1,0].get_ylim())
        updated = True
      fig.canvas.draw()

  #def on_xlim_change(*args):
  #  #nonlocal updated
  #  print("here")
  #  print(plt.get_current_fig_manager().toolbar.mode)
  #  if(plt.get_current_fig_manager().toolbar.mode == ' '):
  #    updated = False

  width1 = kwargs.get('width1',4.0); width2 = kwargs.get('width2',4.0); width3 = kwargs.get('width3',4.0)
  widths=[width1,width3]
  heights=[width3,width2]
  gs_kw=dict(width_ratios=widths,height_ratios=heights)
  fig,ax=plt.subplots(2,2,figsize=(width1+width3,width2+width3),gridspec_kw=gs_kw)
  plt.subplots_adjust(wspace=0,hspace=0)
  fig.canvas.mpl_connect('key_press_event', key_event)
  fig.canvas.mpl_connect('button_press_event', onclick)
  fig.canvas.mpl_connect('draw_event', ondraw)
  #ax[1,0].callbacks.connect('xlim_changed',on_xlim_change)

  title = kwargs.get('title',' ')
  ax[0,1].text(0.5,0.5,title[curr_pos],horizontalalignment='center',verticalalignment='center',fontsize=50)

  ## xz plane
  slc = bytes2float(data[curr_pos,:,i2,:]) if byte else data[curr_pos,:,i2,:]
  ax[1,0].imshow(slc,interpolation=kwargs.get('interp','none'),aspect='auto',
      extent=[os[0],os[0]+(ns[0])*ds[0],os[2]+ds[2]*(ns[2]),os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
  ax[1,0].tick_params(labelsize=kwargs.get('ticksize',14))
  ax[1,0].plot(loc1*np.ones((ns[2],)),x3,c='k')
  ax[1,0].plot(x1,loc3*np.ones((ns[0],)),c='k')
  ax[1,0].set_xlabel(label1,fontsize=kwargs.get('labelsize',14))
  ax[1,0].set_ylabel(label3,fontsize=kwargs.get('labelsize',14))

  # yz plane
  slc = bytes2float(data[curr_pos,:,:,i1]) if byte else data[curr_pos,:,:,i1]
  im = ax[1,1].imshow(slc,interpolation=kwargs.get('interp','none'),aspect='auto',
      extent=[os[1],os[1]+(ns[1])*ds[1],os[2]+(ns[2])*ds[2],os[2]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
  ax[1,1].tick_params(labelsize=kwargs.get('ticksize',14))
  ax[1,1].plot(loc2*np.ones((ns[2],)),x3,c='k')
  ax[1,1].plot(x2,loc3*np.ones((ns[1],)),c='k')
  ax[1,1].get_yaxis().set_visible(False)
  ax[1,1].set_xlabel(label2,fontsize=kwargs.get('labelsize',14))
  ax1=ax[1,1].twinx()
  ax1.set_ylim(ax[1,1].get_ylim())
  ax1.set_yticks([loc3])
  ax1.set_yticklabels(['%.2f'%(loc3)],rotation='vertical',va='center')
  ax1.tick_params(labelsize=kwargs.get('ticksize',14))
  ax2=ax[1,1].twiny()
  ax2.set_xlim(ax[1,1].get_xlim())
  ax2.set_xticks([loc2])
  ax2.set_xticklabels(['%.2f'%(loc2)])
  ax2.tick_params(labelsize=kwargs.get('ticksize',14))

  # xy plane
  slc = bytes2float(data[curr_pos,i3,:,:]) if byte else data[curr_pos,i3,:,:]
  ax[0,0].imshow(np.flip(slc,0),interpolation=kwargs.get('interp','none'),aspect='auto',
      extent=[os[0],os[0]+(ns[0])*ds[0],os[1],os[1]+(ns[1])*ds[1]],vmin=vmin,vmax=vmax,cmap=kwargs.get('cmap','gray'))
  ax[0,0].tick_params(labelsize=kwargs.get('ticksize',14))
  ax[0,0].plot(loc1*np.ones((ns[1],)),x2,c='k')
  ax[0,0].plot(x1,loc2*np.ones((ns[0],)),c='k')
  ax[0,0].set_ylabel(label2,fontsize=kwargs.get('labelsize',14))
  ax[0,0].get_xaxis().set_visible(False)
  ax3=ax[0,0].twinx()
  ax3.set_ylim(ax[0,0].get_ylim())
  ax3.set_yticks([loc2])
  ax3.set_yticklabels(['%.2f'%(loc2)],rotation='vertical',va='center')
  ax3.tick_params(labelsize=kwargs.get('ticksize',14))
  ax4=ax[0,0].twiny()
  ax4.set_xlim(ax[0,0].get_xlim())
  ax4.set_xticks([loc1])
  ax4.set_xticklabels(['%.2f'%(loc1)])
  ax4.tick_params(labelsize=kwargs.get('ticksize',14))

  # Color bar
  if(kwargs.get('cbar',False)):
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([kwargs.get('barx',0.91),kwargs.get('barz',0.11),
      kwargs.get('wbar',0.02),kwargs.get('hbar',0.78)])
    cbar = fig.colorbar(im,cbar_ax,format='%.2f')
    cbar.ax.tick_params(labelsize=kwargs.get('ticksize',14))
    cbar.set_label(kwargs.get('barlabel',''),fontsize=kwargs.get("barlabelsize",13))
    cbar.draw_all()

  ax[0,1].axis('off')
  if(show):
    plt.show()

def resangframes(resang,dz,dx,dro,oro,jx=10,transp=False,show=True,**kwargs):
  """
  Show a spatial angle gather plot for each rho. Assumes
  input has shape of [nro,nx,na,nz]

  Parameters
    resang - the input residually-migrated angle gathers
    dz     - depth sampling
    dx     - spatial sampling
    dro    - rho sampling
    jx     - subsampling along the x axis (image points to skip) [10]
    transp - flag indicating that the input image has shape [nro,na,nz,nx] [False]
  """
  if(transp):
    # [nro,na,nz,nx] -> [nro,nx,na,nz]
    resangt = np.transpose(resang,(0,3,1,2))
  else:
    resangt = resang
  nz = resangt.shape[3]; na = resangt.shape[2]; nx = resangt.shape[1]; nro = resangt.shape[0]
  # Subsample the spatial axis
  resangts = resangt[:,::jx,:,:]
  nxs = resangts.shape[1]
  # Reshape to flatten the angle and CDP axes
  resangts = resangts.reshape([nro,na*nxs,nz])
  # Plot frames
  viewimgframeskey(resangts,ottl=oro,dttl=dro,ttlstring=r'$\rho$=%.3f',
      pclip=kwargs.get('pclip',1.0),labelsize=kwargs.get('labelsize',14),
      ticksize=kwargs.get('ticksize',14),interp=kwargs.get('interp','none'),
      xmax=kwargs.get('xmax',nx*dx),zmax=kwargs.get('zmax',nz*dz),show=show,
      xlabel='X (km)',ylabel='Z (km)')

def viewresangptch(img,prb,oro,dro,smb=None,streamer=True,fast=True,show=True,**kwargs):
  """
  An interactive visualization of the focused image classification of residual
  migration images

  Parameters:
    img       - the input residually migrated angle patch [nro,na,nz,nx]
    prb       - predicted focus probabilities
    oro       - origin of residual migration axis
    dro       - sampling of residual migration axis
    smb       - computed semblances [None]
    streamer  - streamer acqusition geometry [True]
    vmin      - minimum value to display in the data [default is minimum amplitude of all data]
    vmax      - maximum value to display in the data [default is maximum amplitude of all data]
    pclip     - how much to clip the min and max of the amplitudes [0.9]
    ttlstring - title to be printed. Can be printed of the form ttlstring%(ottl + dttl*(framenumber))
    ottl      - origin for printing title values [0.0]
    dttl      - sampling for printing title values [1.0]
    ttlvals   - metric values to print on the title [None]
    interp    - interpolation type for better display of the data (sinc for seismic, bilinear of velocity) [none]
    show      - flag for calling plt.show() [True]
  """
  if(len(img.shape) < 4):
    raise Exception("Data must be 4D")

  # Compute the stack
  stk = np.sum(img,axis=1)
  # Extract the angle gather from the middle of the patch
  nro,na,nz,nx = img.shape
  if(streamer):
    ang = img[:,32:,:,nx//2]
  else:
    ang = img[:,:,:,nx//2]
  ang = np.transpose(ang,(0,2,1))

  # Normalize predictions and semblance
  prb /= np.max(prb)
  if(smb is not None):
    smb /= np.max(smb)

  # Compute rhos
  rhos = np.linspace(oro,oro+(nro-1)*dro,nro)

  curr_pos = 0
  vmin = kwargs.get('vmin',None); vmax = kwargs.get('vmax',None)
  if(vmin == None or vmax == None):
    svmin = np.min(stk)*kwargs.get('pclip',1.0)
    svmax = np.max(stk)*kwargs.get('pclip',1.0)
    avmin = np.min(ang)*kwargs.get('pclip',1.0)
    avmax = np.max(ang)*kwargs.get('pclip',1.0)

  def key_event(e):
    nonlocal curr_pos,vmin,vmax

    if e.key == "n":
        curr_pos = curr_pos + 1
    elif e.key == "m":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % img.shape[0]

    istk = stk[curr_pos,:,:]
    iang = ang[curr_pos,:,:]
    iprb = prb[curr_pos]
    ax[0].set_title(r'$\rho$=%f prd=%g'%(oro+curr_pos*dro,iprb),fontsize=kwargs.get('labelsize',14))
    if(smb is not None):
      ax[1].set_title(r'smb=%g'%(smb[curr_pos]),fontsize=kwargs.get('labelsize',14))
    ax[0].tick_params(labelsize=kwargs.get('ticksize',14))
    if(fast):
      l1.set_data(istk)
      l2.set_data(iang)
    fig.canvas.draw()

  fig,ax = plt.subplots(1,3,figsize=(kwargs.get("wbox",15),kwargs.get("hbox",6)))
  fig.canvas.mpl_connect('key_press_event', key_event)
  # Show the first frame
  istk = stk[0,:,:]; iang = ang[0,:,:]; iprb = prb[0]
  l1 = ax[0].imshow(istk,cmap=kwargs.get('cmap','gray'),vmin=vmin,vmax=vmax,
                    extent=[kwargs.get('xmin',0.0),kwargs.get('xmax',img.shape[1]),
                    kwargs.get('zmax',img.shape[0]),kwargs.get('zmin',0.0)],
                     interpolation=kwargs.get('interp','bilinear'),aspect=1)
  ax[0].set_xlabel('X (km)',fontsize=kwargs.get('labelsize',14))
  ax[0].set_ylabel('Z (km)',fontsize=kwargs.get('labelsize',14))
  ax[0].tick_params(labelsize=kwargs.get('ticksize',14))
  ax[0].set_title(r'$\rho=$%.4f prd=%g'%(oro,iprb),fontsize=kwargs.get('labelsize',14))
  l2 = ax[1].imshow(iang,cmap=kwargs.get('cmap','gray'),vmin=vmin,vmax=vmax,
                    extent=[kwargs.get('amin',0.0),kwargs.get('amax',img.shape[1]),
                    kwargs.get('zmax',img.shape[0]),kwargs.get('zmin',0.0)],
                    interpolation=kwargs.get('interp','bilinear'),aspect=1)
  ax[1].set_xlabel(r'Angle ($\degree$)',fontsize=kwargs.get('labelsize',14))
  ax[1].set_ylabel('Z (km)',fontsize=kwargs.get('labelsize',14))
  ax[1].tick_params(labelsize=kwargs.get('ticksize',14))
  if(smb is not None):
    ax[1].set_title('smb=%g'%(smb[curr_pos]),fontsize=kwargs.get('labelsize',14))
  ax[2].plot(rhos,prb)
  if(smb is not None):
    ax[2].plot(rhos,smb)

  if(show):
    plt.show()

def qc_f3data(dat,srcx,recx,srcy,recy,nrec,migslc,bidx=0,ntw=1500,dt=0.002,
              dx=25,dy=25,ox=469800,oy=6072350,show=True,**kwargs):
  """
  QCs the F3 data

  Parameters:
    dat  - the input f3 data [ntr,nt]
    srcx - the source x coordinates [nsht]
    recx - the receiver x coordinates [ntr]
    srcy - the source y coordinates [nsht]
    recy - the receiver x coordinates [ntr]
    nrec - the number of receivers [nsht]
  """
  nsht = len(nrec)
  if(nsht != len(srcx) and nsht != len(srcy)):
    raise Exception("Source X/Y coordinates must be same length as nrec")

  ntr = len(recx)
  if(dat.shape[0] != ntr):
    raise Exception("Data must have same number of traces as receiver coordinates")

  pclip = kwargs.get('pclip',1.0)
  dmin = kwargs.get('dmin',np.min(dat))*pclip
  dmax = kwargs.get('dmax',np.max(dat))*pclip

  # Windowed grid
  oxw = ox + 200*dx; oyw = oy + 5*dy
  nxw1,nyw1 = migslc.shape

  # Plotting axes
  oxp,oyp = oxw*0.001,oyw*0.001
  dxp,dyp = dx*0.001,dy*0.001

  # Scale the source and receiver coordinates
  srcx *= 0.001; recx *= 0.001
  srcy *= 0.001; recy *= 0.001

  curr_pos,beg,end = 0,nrec[0],nrec[1]

  def key_event(e):
    nonlocal curr_pos,beg,end

    if e.key == "n":
      curr_pos = curr_pos + 1
      curr_pos = curr_pos % nsht
      if(curr_pos == 0):
        beg = 0
        end = beg
      beg = end
      end += nrec[curr_pos]
    elif e.key == 'y':
      curr_pos = curr_pos + kwargs.get('sjump',125)
      curr_pos = curr_pos % nsht
      if(curr_pos == 0):
        beg = 0
        end = beg
      beg  += sum(nrec[curr_pos-kwargs.get('sjump',125):curr_pos])
      end  = beg + nrec[curr_pos]
    elif e.key == "m":
      curr_pos = curr_pos - 1
      curr_pos = curr_pos % nsht
      if(curr_pos == nsht-1):
        end = ntr
        beg = end
      end = beg
      beg -= nrec[curr_pos]
    elif e.key == 'u':
      curr_pos = curr_pos - kwargs.get('sjump',125)
      curr_pos = curr_pos % nsht
      if(curr_pos == nsht-1):
        end = ntr
        beg = end
      #print(curr_pos,curr_pos+kwargs.get('sjump',125),nrec[curr_pos:curr_pos+kwargs.get('sjump',125)])
      end -= (sum(nrec[curr_pos+1:curr_pos+1+kwargs.get('sjump',125)]))
      beg = end - nrec[curr_pos]
    else:
        return
    #print(beg,end,curr_pos,nrec[curr_pos])

    # Update the data
    ax[0].set_title('Srcx=%.3f Srcy=%.3f Num=%d/%d'%(srcx[curr_pos],srcy[curr_pos],curr_pos,nsht),fontsize=kwargs.get('labelsize',14))
    ax[0].set_xlabel(kwargs.get('xlabel','X (km)'),fontsize=kwargs.get('labelsize',14))
    ax[0].set_ylabel(kwargs.get('ylabel','Y (km)'),fontsize=kwargs.get('labelsize',14))
    ax[0].tick_params(labelsize=kwargs.get('ticksize',14))
    # Update sources
    srcs = np.zeros([1,2])
    srcs[:,0] = srcx[curr_pos]; srcs[:,1] = srcy[curr_pos]
    scats.set_offsets(srcs)
    # Update receivers
    recs = np.zeros([nrec[curr_pos],2])
    recs[:,0] = recx[beg:end]; recs[:,1] = recy[beg:end]
    scatr.set_offsets(recs)
    # Update the shot gathers
    l.set_data(dat[beg:end,0:ntw].T)
    fig.canvas.draw()

  fig,ax = plt.subplots(2,1,figsize=(kwargs.get("wbox",12),kwargs.get("hbox",8)))
  fig.canvas.mpl_connect('key_press_event', key_event)
  # Show the first frame
  ax[0].imshow(np.flipud(migslc.T),extent=[oxp,oxp+nxw1*dxp,oyp,oyp+nyw1*dyp],
            interpolation='none',cmap='gray')
  scats = ax[0].scatter(srcx[0],srcy[0],marker='*',color='tab:red')
  scatr = ax[0].scatter(recx[0:nrec[0]],recy[0:nrec[0]],marker='v',color='tab:green')
  ax[0].set_xlabel(kwargs.get('xlabel','X (km)'),fontsize=kwargs.get('labelsize',14))
  ax[0].set_ylabel(kwargs.get('ylabel','Y (km)'),fontsize=kwargs.get('labelsize',14))
  ax[0].tick_params(labelsize=kwargs.get('ticksize',14))
  l = ax[1].imshow(dat[0:nrec[0],0:ntw].T,vmin=dmin,vmax=dmax,extent=[0,nrec[0],ntw*dt,0],
                   interpolation='bilinear',cmap='gray',aspect=kwargs.get('aspect',50))
  ax[1].set_xlabel('Receiver number',fontsize=kwargs.get('labelsize',14))
  ax[1].set_ylabel('Time (s)',fontsize=kwargs.get('labelsize',14))
  if(show):
    plt.show()


def vol_slider(cub, **kwargs) -> None:
  """
  A volume slider using vedo

  Parameters:
     cub - the input cube
  """
  pass
