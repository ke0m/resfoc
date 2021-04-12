import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt
from scaas.off2ang import off2angkzx, get_angkzx_axis

sep = seppy.sep()

iaxes,img = sep.read_file("spimgextfrancdistr_2.H")
img = img.reshape(iaxes.n,order='F').T
imgn = img[np.newaxis]
[dz,dx,dy,dhx] = iaxes.d; [oz,ox,oy,ohx] = iaxes.o

na = 64
ang = off2angkzx(imgn,ohx,dhx,dz,eps=1.0,na=na,transp=True)
#na = 41
#ang = off2angkzx(imgn,ohx,dhx,dz,na=na,transp=True)

na,oa,da = get_angkzx_axis(na=na)

sep.write_file("spimgfrancang_2.H",ang.T,os=[0,oa,0.0,ox,0.0],ds=[dz,da,1.0,dx,1.0])

