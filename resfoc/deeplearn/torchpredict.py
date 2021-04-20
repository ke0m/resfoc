"""
Functions for making predictions with torch models

@author: Joseph Jennings
@version: 2020.09.21
"""
import numpy as np
import torch
from deeplearn.utils import thresh, normalize, resizepow2
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from scaas.trismooth import smooth

def segmentfaults(img,net,nzp=128,nxp=128,strdz=None,strdx=None,resize=False):
  """
  Segments faults on a 2D image. Returns the probablility of each
  pixel being a fault or not.

  Parameters:
    img    - the input image [nz,nx]
    net    - the torch network with trained weights
    nzp    - z-dimension of the patch provided to the CNN [128]
    nxp    - x-dimension of the patch provided to the CNN [128]
    strdz  - z-dimension of the patch stride (50% overlap) [npz/2]
    strdx  - x-dimension of the patch stride (50% overlap) [npx/2]
    resize - option to resize the image to a power of two in each dimension [False]
    verb   - verbosity flag [False]

  Returns the spatial fault probability map [nz,nx]
  """
  # Resample to nearest power of 2
  if(resize):
    rimg = resizepow2(img,kind='linear')
  else:
    rimg = img
  # Perform the patch extraction
  if(strdz is None): strdz = int(nzp/2)
  if(strdx is None): strdx = int(nxp/2)
  pe = PatchExtractor((nzp,nxp),stride=(strdx,strdz))
  iptch = pe.extract(rimg)
  numpz = iptch.shape[0]; numpx = iptch.shape[1]
  iptch = iptch.reshape([numpx*numpz,1,nzp,nxp])
  # Normalize each patch
  niptch = np.zeros(iptch.shape)
  for ip in range(numpz*numpx):
    niptch[ip,:,:] = normalize(iptch[ip,:,:])
  # Convert to torch tensor
  tniptch = torch.from_numpy(niptch.astype('float32'))
  # Make a prediction
  with torch.no_grad():
    iprd  = torch.sigmoid(net(tniptch)).numpy()
  # Reconstruct the predictions
  ipra  = iprd.reshape([numpz,numpx,nzp,nxp])
  iprb  = pe.reconstruct(ipra)

  return iprb

