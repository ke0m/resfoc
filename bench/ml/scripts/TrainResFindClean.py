"""
Trains a CNN for selecting which residual migration
image is the closest to the true, well-focused image

The architecture is an encoder that goes into a dense layer
and then into a softmax layer

The input training data must be in H5 format. This program writes
out the model architecture as well as the model weights
and biases which can then be used subsequently for validation/testing.

@author: Joseph Jennings
@version: 2020.04.30
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import h5py
from deeplearn.dataloader import load_allssimcleandata
from deeplearn.kerasnets import findres
from deeplearn.kerascallbacks import ShowPred
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "y",
    "nepochs": 10,
    "drpout": 0.0,
    "gpus": []
    }
if args.conf_file:
  config = configparser.ConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

# Set defaults
parser.set_defaults(**defaults)

# Input files
ioArgs = parser.add_argument_group('Inputs and outputs')
ioArgs.add_argument("-trdat",help="Input training dataset in H5 format",type=str,required=True)
ioArgs.add_argument("-vadat",help="Input validation dataset in H5 format",type=str,required=True)
ioArgs.add_argument("-wgtout",help="Output CNN filter coefficients",type=str)
ioArgs.add_argument("-arcout",help="Output CNN architecture",type=str)
ioArgs.add_argument("-chkpnt",help="Model checkpoint file",type=str)
ioArgs.add_argument("-lssout",help="Output loss history",type=str)
ioArgs.add_argument("-vlsout",help="Output validation loss history",type=str)
ioArgs.add_argument("-accout",help="Output accuracy history",type=str)
ioArgs.add_argument("-vacout",help="Output validation accuracy history",type=str)
# Test data
tstArgs = parser.add_argument_group("Paths for analyzing test data predictions")
tstArgs.add_argument("-imgin",help="Input well focused test image",type=str)
tstArgs.add_argument("-resin",help="Input residual migration images",type=str)
tstArgs.add_argument("-ptbin",help="Input velocity perturbation",type=str)
# Training
trainArgs = parser.add_argument_group('Training parameters')
trainArgs.add_argument('-bsize',help='Batch size [20]',type=int)
trainArgs.add_argument('-nepochs',help='Number of passes over training data [10]',type=int)
# Other arguments
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
parser.add_argument("-gpus",help="A comma delimited list of which GPUs to use [default all]",type=str)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# Get command line arguments
verb  = sep.yn2zoo(args.verb)
gpus  = sep.read_list(args.gpus,[])
if(len(gpus) != 0):
  for igpu in gpus: os.environ['CUDA_VISIBLE_DEVICES'] = str(igpu)

# Training arguments
bsize   = args.bsize
nepochs = args.nepochs

# Load all data
#allx,ally = load_allssimdata(args.trdat,None,105)
#allx,ally = load_allssimdata(args.trdat,None,75)
allx,ally = load_allssimcleandata(args.trdat,None)
xshape = allx.shape[1:]
yshape = ally.shape[1:]

model = findres()

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Create callbacks
checkpointer = ModelCheckpoint(filepath=args.chkpnt, verbose=1, save_best_only=True)
predicter    = ShowPred(args.imgin,args.resin,args.ptbin,psize=(19,64,64),ssize=(19,32,32))

# Train the model
history = model.fit(allx,ally,epochs=nepochs,batch_size=bsize,verbose=1,shuffle=True,
                   validation_split=0.2,callbacks=[checkpointer,predicter])

# Write the model
model.save_weights(args.wgtout)

# Save the loss history
lossvepch = np.asarray(history.history['loss'])
sep.write_file(args.lssout,lossvepch)
vlssvepch = np.asarray(history.history['val_loss'])
sep.write_file(args.vlsout,vlssvepch)

# Save the accuracy history
accvepch = np.asarray(history.history['acc'])
sep.write_file(args.accout,accvepch)
vacvepch = np.asarray(history.history['val_acc'])
sep.write_file(args.vacout,vacvepch)

# Save the model architecture
with open(args.arcout,'w') as f:
  f.write(model.to_json())
