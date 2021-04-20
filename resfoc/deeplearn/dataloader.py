"""
Classes and functions for loading in data for deep learning

@author: Joseph Jennings
@version: 2020.10.29
"""
import os as opsys
import h5py
import numpy as np
import random
import subprocess
from genutils.ptyprint import progressbar, create_inttag

class WriteToH5:

  def __init__(self,name,nmax=1000000,dsize=1,channels_last=False):
    """
    Creates a WriteToH5 object that writes training data
    to an HDF5 file

    Parameters:
      name           - the name of the output HDF5 file
      dsize          - number of training examples in an individual H5 dataset [1]
      channels_last  - flag indicating channels should come last [False]

    Returns WriteToH5 object
    """
    self.__hf = h5py.File(name,'w',libver='latest')
    self.__dsize = dsize
    self.__channels_last = False
    # For naming the datasets
    self.__nmax = nmax; self.__ctr = 0
    # Left over examples
    self.__xlef = []; self.__ylef = []
    self.__recurse = False

  def write_examples(self,x,y):
    """
    Writes training examples to the H5 file

    Parameters:
      x - the training examples
      y - the corresponding labels
    """
    tag = create_inttag(self.__ctr,self.__nmax)
    if(x.shape[0] != y.shape[0]):
      raise Exception("Number of examples must be same for input data and labels")
    nex = x.shape[0]

    # Shapes of each example
    if(self.__channels_last):
      xshape  = [self.__dsize,*x.shape[1:],1]
      yshape  = [self.__dsize,*y.shape[1:],1]
    else:
      xshape  = [self.__dsize,1,*x.shape[1:]]
      yshape  = [self.__dsize,1,*y.shape[1:]]

    igr,rem = divmod(nex,self.__dsize)

    if(igr > 0):
      # Write out what fits
      beg = 0; end = 0
      for k in range(igr):
        beg = end; end += self.__dsize
        datatag = create_inttag(self.__ctr,self.__nmax)
        if(self.__channels_last):
          xot = np.expand_dims(x[beg:end],axis=-1)
          yot = np.expand_dims(y[beg:end],axis=-1)
        else:
          xot = x[beg:end,np.newaxis]
          yot = y[beg:end,np.newaxis]
        self.__hf.create_dataset('x'+datatag,xshape,data=xot,dtype=np.float32)
        self.__hf.create_dataset('y'+datatag,yshape,data=yot,dtype=np.float32)
        self.__ctr += 1
      # Save what you can
      if(end < nex):
        if(self.__recurse):
          self.__xlef = []
          self.__ylef = []
          self.__recurse = False
        self.__xlef.append(x[end:])
        self.__ylef.append(y[end:])
      elif(end == nex):
        if(self.__recurse):
          self.__recurse = False
          self.__xlef = []
          self.__ylef = []
    else:
      # Append the examples to the saved list
      self.__xlef.append(x)
      self.__ylef.append(y)

    # If the size of the left over array is larger than dsize
    # and recurse
    nlef = np.sum([ex.shape[0] for ex in self.__ylef])
    if(nlef >= self.__dsize):
      xr = np.concatenate(self.__xlef,axis=0)
      yr = np.concatenate(self.__ylef,axis=0)
      self.__recurse = True
      self.write_examples(xr,yr)

  def __del__(self):
    """
    Deletes a WriteToH5 object
    """
    try:
      self.__hf.close()
    except:
      pass

def splith5(fin,f1,f2,split=0.8,rand=False,clean=True):
  """
  Splits an H5 file into two other files.
  Useful for splitting data into training and validation
  """
  hfin = h5py.File(fin,'r')
  hf1  = h5py.File(f1,'w')
  hf2  = h5py.File(f2,'w')
  keys = list(hfin.keys())
  nb   = int(len(keys)/2)
  nf1 = int(split*nb)
  nf2 = nb - nf1
  if(rand):
    choices = list(range(nb))
    idxs = random.sample(choices,nf1)
  else:
    idxs = list(range(nf1))
  for idx in progressbar(range(nb), "nbatches:"):
    if idx in idxs:
      hfin.copy(keys[idx],hf1)
      hfin.copy(keys[idx+nb],hf1)
    else:
      hfin.copy(keys[idx],hf2)
      hfin.copy(keys[idx+nb],hf2)
  hfin.close()
  hf1.close()
  hf2.close()
  # Remove the original file
  if(clean):
    sp = subprocess.check_call('rm %s'%(fin),shell=True)

def load_alldata(trfile,vafile,dsize,begex=None,endex=None,verb=False):
  """ Loads all data and labels into numpy arrays """
  # Get training number of examples
  hftr = h5py.File(trfile,'r')
  trkeys = list(hftr.keys())
  ntr = int(len(trkeys)/2)
  # Get the validation number of examples
  if(vafile != None):
    hfva = h5py.File(vafile,'r')
    vakeys = list(hfva.keys())
    nva = int(len(vakeys)/2)
  else:
    nva = 0; vakeys = []
  # Get shape of examples
  xshape = hftr[trkeys[0]].shape
  yshape = hftr[trkeys[0+ntr]].shape
  # Get number of examples
  if(begex is None or endex is None):
    begex = 0; endex = ntr
    nex = ntr
  else:
    nex = endex - begex
  # Allocate output arrays
  allx = np.zeros([(nex+nva)*dsize,*xshape[1:]],dtype='float32')
  ally = np.zeros([(nex+nva)*dsize,*yshape[1:]],dtype='float32')
  k = 0
  # Get all training examples
  for itr in progressbar(range(begex,endex), "numtr:",verb=verb):
    for iex in range(dsize):
      allx[k]  = hftr[trkeys[itr]    ][iex]
      ally[k]  = hftr[trkeys[itr+ntr]][iex]
      k += 1
  # Get all validation examples
  for iva in progressbar(range(nva), "numva:"):
    for iex in range(dsize):
      allx[k]  = hfva[vakeys[iva]    ][iex]
      ally[k]  = hfva[vakeys[iva+nva]][iex]
      k += 1
  # Close the files
  hftr.close()
  if(vafile != None): hfva.close()

  return allx,ally

def load_all_unlabeled_data(filein,begex=None,endex=None):
  """ Loads all data into a numpy array """
  # Get training number of examples
  hftr = h5py.File(filein,'r')
  trkeys = list(hftr.keys())
  ntr = int(len(trkeys))
  # Get shape of examples
  xshape = hftr[trkeys[0]].shape
  # Allocate output arrays
  allx = []
  k = 0
  # Get all training examples
  if(begex is None or endex is None):
    begex = 0; endex = ntr
  for itr in progressbar(range(begex,endex), "numtr:"):
    dsize = hftr[trkeys[itr]].shape[0]
    for iex in range(dsize):
      allx.append(hftr[trkeys[itr]][iex])
      k += 1
  # Close the file
  hftr.close()
  return np.asarray(allx)

def load_allflddata(fldfile,dsize):
  """ Loads all field (unlabeled) data into a numpy array """
  # Get training number of examples
  hffld = h5py.File(fldfile,'r')
  fldkeys = list(hffld.keys())
  nfld = int(len(fldkeys)/2)
  # Get shape of examples
  xshape = hffld[fldkeys[0]].shape
  # Allocate output arrays
  if(len(xshape) == 4):
    allx = np.zeros([(nfld)*dsize,xshape[1],xshape[2],xshape[3]],dtype='float32')
  elif(len(xshape) == 3):
    allx = np.zeros([(nfld)*dsize,xshape[1],xshape[2]],dtype='float32')
  k = 0
  # Get all field examples
  for ifld in progressbar(range(nfld), "numfld:"):
    for iex in range(dsize):
      allx[k,:,:,:]  = hffld[fldkeys[ifld]][iex,:,:,:]
      k += 1

  return allx

def load_allpatchdata(trfile,vafile,dsize):
  """
  Loads all data and labels into numpy arrays

  Works both for the SSIM residual migration training
  and the patchwise fault classification training
  """
  # Get training number of examples
  hftr = h5py.File(trfile,'r')
  trkeys = list(hftr.keys())
  ntr = int(len(trkeys)/2)
  # Get the validation number of examples
  if(vafile != None):
    hfva = h5py.File(vafile,'r')
    vakeys = list(hfva.keys())
    nva = int(len(vakeys)/2)
  else:
    nva = 0; vakeys = []
  # Get shape of examples
  xshape = hftr[trkeys[0]].shape
  yshape = hftr[trkeys[0+ntr]].shape
  # Allocate output arrays
  if(len(xshape) == 4):
    allx = np.zeros([(ntr+nva)*dsize,xshape[1],xshape[2],xshape[3]],dtype='float32')
  ally = np.zeros([(ntr+nva)*dsize,yshape[1]],dtype='float32')
  k = 0
  # Get all training examples
  for itr in progressbar(range(ntr), "numtr:"):
    for iex in range(dsize):
      allx[k,:,:,:]  = hftr[trkeys[itr]    ][iex,:,:,:]
      ally[k,:]  = hftr[trkeys[itr+ntr]][iex,:]
      k += 1
  # Get all validation examples
  for iva in progressbar(range(nva), "numva:"):
    for iex in range(dsize):
      allx[k,:,:,:]  = hfva[vakeys[iva]    ][iex,:,:,:]
      ally[k,:]  = hfva[vakeys[iva+nva]][iex,:]
      k += 1
  # Close the files
  hftr.close()
  if(vafile != None): hfva.close()

  return allx, ally

def load_unlabeled_flat_data(filein,begex=None,endex=None):
  """ Loads in flattened unlabeled data """
  hftr = h5py.File(filein,'r')
  trkeys = list(hftr.keys())
  ntr = int(len(trkeys))
  xshape = hftr[trkeys[0]].shape
  if(begex is None or endex is None):
    begex = 0; endex = ntr
    allx = np.zeros([ntr,xshape[0],xshape[1],xshape[2]],dtype='float32')
  else:
    nex = endex - begex
    allx = np.zeros([nex,xshape[0],xshape[1],xshape[2]],dtype='float32')
  for itr in progressbar(range(begex,endex), "numex:"):
    allx[itr,:,:,:]  = hftr[trkeys[itr]][:]
  # Close H5 file
  hftr.close()

  return allx

def load_labeled_flat_data(trfile,vafile,begex=None,endex=None):
  """ Loads flattened labeled data into numpy arrays """
  # Get training number of examples
  hftr = h5py.File(trfile,'r')
  trkeys = list(hftr.keys())
  ntr = int(len(trkeys)/2)
  # Get the validation number of examples
  if(vafile is not None):
    hfva = h5py.File(vafile,'r')
    vakeys = list(hfva.keys())
    nva = int(len(vakeys)/2)
  else:
    nva = 0; vakeys = []
  # Get shape of examples
  xshape = hftr[trkeys[0]].shape
  yshape = hftr[trkeys[0+ntr]].shape
  ndim = len(xshape)
  # Get number of examples
  if(begex is None or endex is None):
    begex = 0; endex = ntr
    nex = ntr
  else:
    nex = endex - begex
  if(ndim == 3):
    allx = np.zeros([(nex+nva),xshape[0],xshape[1],xshape[2]],dtype='float32')
  elif(ndim == 4):
    allx = np.zeros([(nex+nva),xshape[0],xshape[1],xshape[2],xshape[3]],dtype='float32')
  ally = np.zeros([(nex+nva),yshape[0]],dtype='float32')
  # Get all training examples
  for itr in progressbar(range(begex,endex), "numtr:"):
    if(ndim == 3):
      allx[itr,:,:,:]  = hftr[trkeys[itr]    ][:]
    elif(ndim == 4):
      allx[itr,:,:,:,:]  = hftr[trkeys[itr]    ][:]
    ally[itr,:]      = hftr[trkeys[itr+ntr]][:]
  # Get all validation examples
  for iva in progressbar(range(nva), "numva:"):
    if(ndim == 3):
      allx[iva,:,:,:]  = hfva[vakeys[iva]    ][:]
    elif(ndim == 4):
      allx[iva,:,:,:,:]  = hfva[vakeys[iva]    ][:]
    ally[iva,:]      = hfva[vakeys[iva+nva]][:]
  # Close the files
  hftr.close()
  if(vafile != None): hfva.close()

  return allx, ally

def clean_examples(ifiles,skips,sep,verb=False) -> None:
  """
  Cleans examples from a .H file

  Parameters:
    ifiles - input list of .H files (assumes all files same shape)
    skips  - file indices over which to skip for cleaning
    verb   - verbosity flag [False]
  """
  if(type(ifiles) is not list):
    raise Exception("Input files must be within a list")
  nf = len(ifiles)

  # Prepare the output files
  ofiles = []
  for ifile in ifiles:
    bname = opsys.path.splitext(ifile)[0]
    ofiles.append(bname + '-cln.H')

  # Get the total enumber of examples
  taxes = sep.read_header(ifiles[0])
  nex = taxes.n[-1]
  os  = taxes.o[:-1]
  ds  = taxes.d[:-1]

  for iex in progressbar(range(nex),"nex:",verb=verb):
    if(iex not in skips):
      for ifile in range(nf):
        axes,dat = sep.read_wind(ifiles[ifile],fw=iex,nw=1)
        dat = dat.reshape(axes.n,order='F')
        if(iex == 0):
          sep.write_file(ofiles[ifile],dat,os=os,ds=ds)
        elif(iex == 1):
          sep.append_file(ofiles[ifile],dat,newaxis=True)
        else:
          sep.append_file(ofiles[ifile],dat)

