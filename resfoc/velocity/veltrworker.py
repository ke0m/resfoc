"""
Worker for creating synthetic velocity models

@author: Joseph Jennings
@version: 2020.08.30
"""
import zmq
from comm.sendrecv import notify_server, send_zipped_pickle, recv_zipped_pickle
from velocity.stdmodels import velfaultsrandom
from scaas.velocity import create_randomptbs_loc
from genutils.ptyprint import progressbar
import numpy as np

# Connect to socket
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://oas.stanford.edu:5555")

# Listen for work from server
while True:
  # Notify we are ready
  notify_server(socket)
  # Get work
  chunk = recv_zipped_pickle(socket)
  # If chunk is empty, keep listening
  if(chunk == {}):
    continue
  # If I received something, do some work
  ochunk = {}
  # Get dimensions
  nmdl = chunk[0]; nx = chunk[1]['nx']; nz = chunk[1]['nz']
  ochunk['vel'] = np.zeros([nmdl,nx,nz],dtype='float32')
  ochunk['ref'] = np.zeros([nmdl,nx,nz],dtype='float32')
  ochunk['cnv'] = np.zeros([nmdl,nx,nz],dtype='float32')
  ochunk['lbl'] = np.zeros([nmdl,nx,nz],dtype='float32')
  if(chunk[2]):
    ochunk['ano'] = np.zeros([nmdl,nx,nz],dtype='float32')
  # Input parameters
  layer = chunk[1]['layer']; maxvel = chunk[1]['maxvel']
  # Loop over all models to create
  for imdl in progressbar(range(nmdl),"nmod:"):
    ochunk['vel'][imdl],ochunk['ref'][imdl],\
      ochunk['cnv'][imdl],ochunk['lbl'][imdl] = velfaultsrandom(**chunk[1],verb=False)
    if(chunk[2]):
      ochunk['ano'][imdl] = create_randomptbs_loc(nz,nx,**chunk[3]).T
  # Return other parameters if desired
  ochunk['cid']  = chunk[3]
  # Tell server this is the result
  ochunk['msg'] = "result"
  # Send back the result
  send_zipped_pickle(socket,ochunk)
  # Receive 'thank you'
  socket.recv()

