import torch
import numpy as np
from sigsbee_focdata import SigsbeeFocData
from sigsbee_focdata_gpu import SigsbeeFocDataGPU
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from deeplearn.utils import torchprogress
from deeplearn.torchnets import Vgg3_3d
from deeplearn.torchlosses import bal_ce
from genutils.ptyprint import create_inttag
from genutils.plot import plot_cubeiso

# Get the GPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Training set
#sig_fdr = SigsbeeFocDataGPU('/net/thing/scr2/joseph29/sigsbee_focdefres.h5',device,verb=True,begex=0,endex=100)
sig_fdr = SigsbeeFocData('/net/thing/scr2/joseph29/sigsbee_focdefres.h5',nexmax=100)

# Get idxs and cutoff
nex   = len(sig_fdr); split = 0.2
idxs  = list(range(nex))
split = int(np.floor(split*nex))

# Shuffle the idxs
seed  = 1992
np.random.seed(seed)
np.random.shuffle(idxs)

# Randomly split examples
trnidx,validx = idxs[split:], idxs[:split]

# Samplers
trsampler = SubsetRandomSampler(trnidx)
vasampler = SubsetRandomSampler(validx)

trloader = DataLoader(sig_fdr,batch_size=20,num_workers=0,sampler=trsampler)
valoader = DataLoader(sig_fdr,batch_size=20,num_workers=0,sampler=vasampler)

# Get the network
net = Vgg3_3d()
net.to(device)

# Loss function
criterion = bal_ce(device)
criterion.to(device)

# Optimizer
lr = 1e-4
optimizer = torch.optim.Adam(net.parameters(),lr=lr)

# Training
nepoch = 10

prds = []; macc = k = 0
for trdat in trloader:
  inputs,labels = trdat['img'].to(device),trdat['lbl'].to(device)
  print(inputs.size(),labels.size())
  if(labels[0].item() == 0):
    print("Defocused")
  else:
    print("Focused")
  #plot_cubeiso(inputs[0,0].cpu().numpy(),stack=True,elev=15,show=True,verb=False)
  outputs = net(inputs)
  iprd = ((outputs > 0.5).float() == labels).float()
  macc += iprd.sum().item()
  k += 20
  print(macc/k)
  prds.append(iprd)
  loss = criterion(outputs,labels)

allprds = torch.cat(prds)

print(macc,allprds.sum().item())

tacc = (allprds.sum().item()/allprds.size()[0])
nacc = (macc/allprds.size()[0])
print(tacc,nacc)
