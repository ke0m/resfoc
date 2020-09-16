import inpout.seppy as seppy
import numpy as np
import h5py
from resfoc.gain import agc
from deeplearn.utils import normextract, resample, plotseglabel
from deeplearn.focuslabels import corrsim, semblance_power
from deeplearn.dataloader import WriteToH5
import matplotlib.pyplot as plt
from genutils.ptyprint import create_inttag, progressbar

# IO
sep = seppy.sep()

iaxes,img = sep.read_file("spimgbobang.H")
[nz,na,ny,nx] = iaxes.n
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)
imgw = img[:,0,:,:]
imgs = np.sum(imgw,axis=1)

# Size of a single patch
ptchz = 64; ptchx = 64


# Define window
bxw = 50; exw = nx - 50
bzw = 0;  ezw = nz

imgsw = imgs[bxw:exw,bzw:ezw]
imgg  = agc(imgsw)
# Transpose
imggt = np.ascontiguousarray(imgg.T)
# Extract patches
iptch = normextract(imggt,nzp=ptchz,nxp=ptchx,norm=True)
nptch = iptch.shape[0]

hf = h5py.File("/scr2/joseph29/hale_fltseg.h5",'r')
keys = list(hf.keys())
nex = len(keys)//2

while True:
  # Get indices
  tsidx = np.random.choice(300)
  tridx = np.random.randint(nex)
  plt.figure()
  plt.imshow(iptch[tsidx],cmap='gray',interpolation='bilinear',aspect=0.5)
  plotseglabel(hf[keys[tridx]][0,:,:,0],hf[keys[tridx+nex]][0,:,:,0],aratio=0.5,interpolation='bilinear',show=True)

hf.close()
