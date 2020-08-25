import inpout.seppy as seppy
import numpy as np
import zmq
from velocity.veltrchunkr import veltrchunkr
from server.distribute import dstr_collect
from client.sshworkers import launch_sshworkers, kill_sshworkers

# IO
sep = seppy.sep()

# Start workers
hosts = ['fantastic', 'thing', 'storm', 'torch']
cfile = "/homes/sep/joseph29/projects/resfoc/velocity/veltrworker.py"
launch_sshworkers(cfile,hosts=hosts,sleep=1,verb=1,clean=False)

# Make generator
nchnk = len(hosts)
vcnkr = veltrchunkr(nchnk,
                    nmodels=10,
                    nx=2133,ny=20,nz=1201,layer=100,maxvel=3800)
gen = iter(vcnkr)

# Bind to socket
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")

# Distribute work to workers and collect results
okeys = ['vel','ref','cnv','lbl']
output = dstr_collect(okeys,nchnk,gen,socket)

ovel = np.concatenate(output['vel'])
oref = np.concatenate(output['ref'])
ocnv = np.concatenate(output['cnv'])
olbl = np.concatenate(output['lbl'])

# Write output
dz = 0.005; dx = 0.01143; ox = 3.05562
sep.write_file('sigsbee_trvels.H',ovel.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0])
sep.write_file('sigsbee_trrefs.H',oref.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0])
sep.write_file('sigsbee_trcnvs.H',ocnv.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0])
sep.write_file('sigsbee_trlbls.H',olbl.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0])

# Clean up
kill_sshworkers(cfile,hosts,verb=False)

