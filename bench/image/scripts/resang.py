import inpout.seppy as seppy
import numpy as np
from scaas.off2ang import off2ang
from resfoc.resmig import preresmig,get_rho_axis
from resfoc.gain import tpow
from utils.movie import viewimgframeskey

sep = seppy.sep()

#iaxes,img = sep.read_file("fltimgextprcnew.H")
iaxes,img = sep.read_file("../doc/spring2020/extimg.H")
img = img.reshape(iaxes.n,order='F')

dz = iaxes.d[0]; dx = iaxes.d[1]; dh = iaxes.d[2]

imgt = np.ascontiguousarray(np.transpose(img,(2,0,1))).astype('float32')
viewimgframeskey(imgt,transp=False)

# First do residual migration
inro = 2; idro = 0.005
storm = preresmig(imgt,[dh,dz,dx],nro=inro,dro=idro,time=False,transp=True,verb=True,nthreads=2)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

nh = storm.shape[1]
viewimgframeskey(storm[:,int(nh/2),:,:],transp=False,ttlstring='rho=%.3f',ottl=ooro,dttl=odro)

raxes = seppy.axes([storm.shape[2],storm.shape[3],storm.shape[1],storm.shape[0]],[0.0,0.0,iaxes.o[2],ooro],[dz,dx,dh,odro])
stormt = np.transpose(storm,(2,3,1,0))
sep.write_file("bigresmigtest2.H",stormt,ofaxes=raxes)

