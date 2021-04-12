import inpout.seppy as seppy
import numpy as np
import zmq
from oway.imagechunkr import imagechunkr
from server.distribute import dstr_sum
from server.utils import startserver
from client.sshworkers import launch_sshworkers, kill_sshworkers

# IO
sep = seppy.sep()

# Start workers
hosts = ['fantastic','storm','torch','thing','jarvis']
cfile = "/homes/sep/joseph29/projects/scaas/oway/imageworker.py"
launch_sshworkers(cfile,hosts=hosts,sleep=1,verb=1,clean=False)

# Read in data
#daxes,dat = sep.read_file("hale_shotflatbob.H")
daxes,dat = sep.read_file("hale_shotflatfranc_2.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
[nt,ntr] = daxes.n; [ot,_] = daxes.o; [dt,_] = daxes.d

# Read in the velocity model
vaxes,vel = sep.read_file("vintzcomb.H")
vel = vel.reshape(vaxes.n,order='F')
[nz,nvx] = vaxes.n; [dz,dvx] = vaxes.d; [oz,ovx] = vaxes.o
ny = 1; dy = 1.0
velin = np.zeros([nz,ny,nvx],dtype='float32')
velin[:,0,:] = vel

# Read in coordinates
saxes,srcx = sep.read_file("hale_srcxflatbob.H")
raxes,recx = sep.read_file("hale_recxflatbob.H")
_,nrec= sep.read_file("hale_nrecbob.H")
nrec = nrec.astype('int')

nxi = int(2*nvx); dxi = dvx/2; oxi = ovx

nchnk = len(hosts)
icnkr = imagechunkr(nchnk,
                    nxi,dxi,ny,dy,nz,dz,velin,
                    dat,dt,minf=1.0,maxf=51.0,
                    nrec=nrec,srcx=srcx,recx=recx,
                    ovx=ovx,dvx=dvx,ox=oxi)

icnkr.set_image_pars(ntx=16,nthrds=40,nrmax=10,sverb=True)
gen = iter(icnkr)

# Bind to socket
context,socket = startserver()

# Distribute work to workers and sum over results
img = dstr_sum('cid','result',nchnk,gen,socket,icnkr.get_img_shape())

# Zero-offset image
sep.write_file("spimgfrancdistr_2.H",img,os=[oz,0.0,oxi],ds=[dz,1.0,dxi])

kill_sshworkers(cfile,hosts,verb=False)

