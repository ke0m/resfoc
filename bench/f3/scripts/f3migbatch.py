import os
import inpout.seppy as seppy
import numpy as np
import time
import oway.coordgeom as geom
from oway.imagechunkr import imagechunkr
from seis.f3utils import compute_batches, compute_batches_var, plot_acq
from server.distribute import dstr_sum, dstr_sum_adapt
from server.utils import startserver, stopserver
from client.slurmworkers import launch_slurmworkers, kill_slurmworkers,\
                                restart_slurmworkers
from genutils.ptyprint import progressbar, create_inttag
from genutils.movie import viewcube3d

# IO
sep = seppy.sep()
qc = False

# Start workers
cfile = "/home/joseph29/projects/scaas/oway/imageworker.py"
logpath = "./log"
wrkrs,status = launch_slurmworkers(cfile,nworkers=50,wtime=120,queue=['sep','twohour'],
                                   block=['maz132'],logpath=logpath,slpbtw=4.0,mode='adapt')
print("Workers status: ",*status)

# Read in the geometry
sxaxes,srcx = sep.read_file("f3_srcx2_700.H")
syaxes,srcy = sep.read_file("f3_srcy2_700.H")
rxaxes,recx = sep.read_file("f3_recx2_700.H")
ryaxes,recy = sep.read_file("f3_recy2_700.H")
naxes,nrec = sep.read_file("f3_nrec2_700.H")
nrec = nrec.astype('int32')
totnsht = len(nrec)

# Read in the windowed velocity model
vaxes,vel = sep.read_file("miglintz5m.H")
vel = np.ascontiguousarray(vel.reshape(vaxes.n,order='F').T)
ny,nx,nz = vel.shape
dz,dx,dy = vaxes.d; oz,ox,oy = vaxes.o

## Window the velocity model
velw = vel[25:125,:700,:1000]
nyw,nxw,nzw = velw.shape
oyw = oy + 25*dy
velwt = np.ascontiguousarray(np.transpose(velw,(2,0,1))) # [ny,nx,nz] -> [nz,ny,nx]
#viewcube3d(velwt,ds=[dz,dx,dy],os=[oz,ox,oyw],cmap='jet',width1=1.0)

# Read in time slice for QC
saxes,slc = sep.read_wind("migwt.T",fw=400,nw=1)
slc = slc.reshape(saxes.n,order='F')
slcw = slc[25:125,:700]

# Output image
img = np.zeros(velwt.shape,dtype='float32') # [nz,ny,nx]

# Compute the batches
#ibatch = 100
#bsize,nb = compute_batches(ibatch,totnsht)
#print("Shot batch size: %d"%(bsize))
ibatch = 450
bsizes = compute_batches_var(ibatch,totnsht)
nb = len(bsizes)
print("Shot batch sizes: ",*bsizes)

# Bind to socket
context,socket = startserver()
trestart = 90*60 # Restart every 90 min

totred,isht = 0,0
for ibtch in progressbar(range(nb),"nbtch",verb=True):
  # Check time elapsed
  if(ibtch == 0): start = time.time()
  elapse = time.time() - start
  if(elapse >= trestart):
    # Stop the server
    stopserver(context,socket)
    # Restart the workers
    status = restart_slurmworkers(wrkrs,limit=False,slpbtw=4.0)
    # Start the server again
    context,socket = startserver()
    # Restart the timing
    start = time.time()

  # Read in the data in batches
  # Window the source geometry
  srcxw = srcx[isht:isht+bsizes[ibtch]]*0.001
  srcyw = srcy[isht:isht+bsizes[ibtch]]*0.001
  nrecw = nrec[isht:isht+bsizes[ibtch]]
  # Compute the number of traces to read in
  nred = np.sum(nrecw)
  # Window the receivers
  recxw = recx[totred:totred+nred]*0.001
  recyw = recy[totred:totred+nred]*0.001
  # Read in the data
  daxes,dat = sep.read_wind("f3_shots2interp_700_muted_debub_onetr_gncw.H",fw=totred,nw=nred)
  dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
  nt,ntr = daxes.n; ot,_ = daxes.o; dt,_ = daxes.d
  isht   += bsizes[ibtch]
  totred += nred

  # Plot the acquisition for QC
  if(qc):
    plot_acq(srcxw,srcyw,recxw,recyw,slcw,ox=ox,oy=oyw,recs=False,show=False)
    plot_acq(srcxw,srcyw,recxw,recyw,slcw,ox=ox,oy=oyw,recs=True,show=True)

  # Check if output image has been written, if so continue
  ofile = "./f3imgs/f3img5m700/f3img5m700-%s.H"%(create_inttag(ibtch,nb))
  if(os.path.exists(ofile)):
    taxes,timg = sep.read_file(ofile)
    timg = np.ascontiguousarray(timg.reshape(taxes.n,order='F')).astype('float32')
    img  = np.ascontiguousarray(np.transpose(timg,(0,2,1))) # [nz,nx,ny] -> [nz,ny,nx]
  else:
    # Build the image chunker (chunks the data)
    #nchnk = bsize//2
    nchnk = status.count('R')
    icnkr = imagechunkr(nchnk,
                        nxw,dx,nyw,dy,nzw,dz,velwt,
                        dat,dt,minf=1.0,maxf=61.0,
                        nrec=nrecw,srcx=srcxw,recx=recxw,
                        srcy=srcyw,recy=recyw,ox=ox,oy=oyw,verb=False)
    icnkr.set_image_pars(ntx=16,nty=16,nhx=0,nrmax=20,nthrds=40,sverb=True,wverb=False)
    gen = iter(icnkr)

    # Distribute work to workers and sum the results
    img += dstr_sum('cid','result',nchnk,gen,socket,icnkr.get_img_shape())
    #img += dstr_sum_adapt('cid','result',nchnk,gen,socket,icnkr.get_img_shape(),
    #                      wrkrs,interval=15,verb=True,logfile='./log/f3imgs.log')
    imgt = np.transpose(img,(0,2,1)) # [nz,ny,nx] -> [nz,nx,ny]

    # Write out current image
    sep.write_file(ofile,imgt,ds=[dz,dx,dy],os=[0.0,ox,oyw])

# Clean up
kill_slurmworkers(wrkrs)

