"""
Keras callbacks for QCing training

@author: Joseph Jennings
@version: 2020.02.21
"""
import inpout.seppy as seppy
import h5py
from tensorflow.keras import callbacks
from deeplearn.dataloader import load_allflddata
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from deeplearn.utils import thresh, plot_seglabel
from resfoc.estro import estro_tgt, onehot2rho
from genutils.ptyprint import create_inttag
import numpy as np
import matplotlib.pyplot as plt

class F3Pred(callbacks.Callback):

  def __init__(self,f3path,predpath,figpath,dsize,psize,ssize):
    # Load in the f3 data
    self.f3dat = load_allflddata(f3path,dsize)
    self.dsize = dsize
    # Create Patch extractor
    self.pe = PatchExtractor(psize,stride=ssize)
    # Dummy array to set dimensions
    self.dummy = np.zeros([512,1024]); dptch = self.pe.extract(self.dummy)
    # Plot one xline
    self.xlidx = 1; self.fs = 200; self.thresh = 0.5
    # Do it only every few epochs
    self.skip = 2
    # Save predictions and figures
    self.predpath = predpath; self.figpath = figpath

  def on_epoch_end(self,epoch,logs={}):
    if(epoch%self.skip == 0):
      # Make a prediction on the f3 data and save it
      print("Predicting on F3 dataset...")
      pred = self.model.predict(self.f3dat,verbose=1)
      # Save predictions to file
      with h5py.File(self.predpath + '/ep%s.h5'%(create_inttag(epoch,100)),"w") as hf:
        hf.create_dataset("pred", self.f3dat.shape, data=pred, dtype=np.float32)
      # Reconstruct a single inline
      iimg = self.f3dat[self.xlidx*self.dsize:(self.xlidx+1)*self.dsize,:,:]
      iimg = iimg.reshape([7,15,128,128])
      rimg = self.pe.reconstruct(iimg)
      # Reconstruct the predictions
      ipred = pred[self.xlidx*self.dsize:(self.xlidx+1)*self.dsize,:,:]
      ipred = ipred.reshape([7,15,128,128])
      rpred = self.pe.reconstruct(ipred)
      # Apply threshold and plot
      tpred = thresh(rpred,self.thresh)
      nt = rimg.shape[0]; nx = rimg.shape[1]
      plotseglabel(rimg[self.fs:,:],tpred[self.fs:,:],color='blue',
          xlabel='Inline',ylabel='Time (s)',xmin=0.0,xmax=(nx-1)*25/1000.0,
          zmin=(self.fs-1)*0.004,zmax=(nt-1)*0.004,vmin=-2.5,vmax=2.5,interp='sinc',aratio=6.5)
      plt.savefig(self.figpath + '/ep%s.png'%(create_inttag(epoch,100)),bbox_inches='tight',dpi=150)


class ShowPred(callbacks.Callback):

  def __init__(self,fpath,rpath,ppath,psize=(19,128,128),ssize=(19,64,64),predpath=None,figpath=None):
    # Initialize SEPlib
    sep = seppy.sep()
    # Residual migration image
    raxes,res = sep.read_file(rpath)
    res = res.reshape(raxes.n,order='F')
    rzro = res[:,:,16,:]
    #rzro = res[psize[1]:,:,16,:] # Remove first row of patches
    [nz,nx,nh,self.nro] = raxes.n; [dz,dx,dh,self.dro] = raxes.d; [ox,ox,oh,self.oro] = raxes.o
    # Perturbation
    paxes,ptb = sep.read_file(ppath)
    self.ptb = ptb.reshape(paxes.n,order='F')
    #self.ptb = self.ptb[psize[1]:,:] # Remove first row of patches
    # Well focused image
    iaxes,img = sep.read_file(fpath)
    img = img.reshape(iaxes.n,order='F')
    izro = img[:,:,16]
    #izro = img[psize[1]:,:,16] # Remove first row of patches
    # Patch it
    self.pe = PatchExtractor(psize,stride=ssize)
    rzrop = np.squeeze(self.pe.extract(rzro.T))
    self.px = rzrop.shape[0]; self.pz = rzrop.shape[1]
    rzrop = rzrop.reshape([self.px*self.pz,self.nro,psize[1],psize[2]])
    self.rzropt = np.transpose(rzrop,(0,2,3,1))
    # Compute ground truth
    self.rho = estro_tgt(rzro.T,izro.T,self.dro,self.oro,nzp=psize[2],nxp=psize[1],strdx=ssize[1],strdz=ssize[2])

  def on_epoch_end(self,epoch,logs={}):
    # Make a prediction on the input image
    print("Predicting on test image...")
    pred = self.model.predict(self.rzropt,verbose=1)
    #print(pred)
    # Apply threshold
    pred = pred.reshape([self.px,self.pz,self.nro])
    print(np.isnan(np.sum(pred)))
    #prho = onehot2rho(pred,self.dro,self.oro,nz=512-128)
    prho = onehot2rho(pred,self.dro,self.oro,nz=512,nzp=64,nxp=64,strdz=32,strdx=32)
    if(epoch%10 == 0):
      fig,ax = plt.subplots(1,3,figsize=(14,7))
      ax[0].imshow(prho.T,cmap='seismic',vmin=0.97,vmax=1.03)
      ax[1].imshow(self.rho.T,cmap='seismic',vmin=0.97,vmax=1.03)
      ax[2].imshow(self.ptb,cmap='jet',vmin=-100,vmax=100)
      plt.savefig("./fig/trainresfind/resfind%d.png"%(epoch),dpi=150,bbox_inches='tight',transparent=True)
      plt.close()

