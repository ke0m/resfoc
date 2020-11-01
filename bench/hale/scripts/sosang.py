import os,sys
from java.nio import *
from java.awt import *
from java.io import *
from java.lang import *
from java.util import *

from edu.mines.jtk.awt import *
from edu.mines.jtk.dsp import *
from edu.mines.jtk.io import *
from edu.mines.jtk.mosaic import *
from edu.mines.jtk.sgl import *
from edu.mines.jtk.util import *
from edu.mines.jtk.util.ArrayMath import *

from sepjy import readFile, writeFile

backgroundColor = Color(0xfd,0xfe,0xff) # easy to make transparent

def main(args):
  nargs = len(sys.argv)
  smps,dat = readFile(sys.argv[1])
  sf,dat = scale(dat)
  # Compute structure tensors
  lof = LocalOrientFilter(4.0)
  s = lof.applyForTensors(dat)
  if(nargs == 4):
    smps,smb = readFile(sys.argv[2])
    smb,datsm = smoothSemblance(smps,dat,s,smb,hw=10)
    datsm = mul(1/sf,datsm)
    # Write smoothed image
    writeFile(sys.argv[3],datsm,smps)
  elif(nargs == 3):
    smb,datsm = smoothSemblance(smps,dat,s,hw=10)
    datsm = mul(1/sf,datsm)
    # Write semblance
    writeFile(makeSmbName(sys.argv[2]),smb,smps)
    # Write smoothed image
    writeFile(sys.argv[2],datsm,smps)

def smoothSemblance(smps,dat,s,smb=None,hw=20,hw2=10):
  if(smb is None):
    # Compute semblance
    lsf = LocalSemblanceFilter(hw,hw)
    smb = lsf.semblance(LocalSemblanceFilter.Direction3.V,s,dat)
    smb = mul(smb,smb)
    smb = mul(smb,smb)
  s.setEigenvalues(0.0,1.0,1.0)
  # Smooth with semblance
  datsm = copy(dat)
  c = hw2*(hw2+1)/6.0
  smooth = LocalSmoothingFilter()
  smooth.apply(s,c,smb,dat,datsm)

  return smb,datsm

def smoothTensors(smps,dat,s,hw2=10):
  # Compute the structure tensors
  d00 = EigenTensors2(s); d00.invertStructure(0.0,0.0)
  d01 = EigenTensors2(s); d01.invertStructure(0.0,1.0)
  d02 = EigenTensors2(s); d02.invertStructure(0.0,2.0)
  d04 = EigenTensors2(s); d04.invertStructure(0.0,4.0)
  d11 = EigenTensors2(s); d11.invertStructure(1.0,1.0)
  d12 = EigenTensors2(s); d12.invertStructure(1.0,2.0)
  d14 = EigenTensors2(s); d14.invertStructure(1.0,4.0)
  datsm = copy(dat)
  c = hw2*(hw2+1)/6.0
  smooth.apply(d04,c,dat,datsm)
  smooth.applySmoothS(datsm,datsm)

  return datsm

def scale(img):
  sf = 1.0
  while(max(img) > 100.0):
    img = mul(img,0.01)
    sf *= 0.01

  return sf,img

def makeSmbName(name):
  return os.path.splitext(name)[0] + '-smb' + os.path.splitext(name)[1]

if __name__ == "__main__":
  main(sys.argv)
