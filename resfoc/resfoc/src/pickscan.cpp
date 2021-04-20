#include <math.h>
#include <cstring>
#include <float.h>
#include <stdio.h>
#include "dynprog.h"

void normalize(int n1, int n2, float *scn) {
  float min = FLT_MAX, max = 0.0;

  for(int i1 = 0; i1 < n1; ++i1) {
    for(int i2 = 0; i2 < n2; ++i2) {
      if(scn[i2*n1 + i1] < min) min = scn[i2*n1 + i1];
      if(scn[i2*n1 + i1] > max) max = scn[i2*n1 + i1];
    }
  }
  if((max-min) < 1e-6) fprintf(stderr,"WARNING: Input semblance range < 1e-6\n");

  for(int i1 = 0; i1 < n1; ++i1) {
    for(int i2 = 0; i2 < n2; ++i2) {
      scn[i2*n1 + i1] = (scn[i2*n1 + i1]-min)/(max-min);
    }
  }
}

void pickscan(float an, int gate, bool norm, float vel0, float o2, float d2,
          int n1, int n2, int n3, float *allscn, float *pck2, float *ampl, float *pcko) {

  /* Depth by midpoint (size of all outputs) */
  int nm = n1*n3;

  /* One scan panel (nz*npar) */
  float *scn = new float[n1*n2]();
  float *wgt = new float[n1*n2](); // Transposed scan panel

  /* Surface velocity */
  int i0 = 0.5 + (vel0-o2)/d2;
  if(i0 <   0) i0 = 0;
  if(i0 >= n2) i0 = n2-1;

  /* Setup picker */
  dynprog dp = dynprog(n1,n2,gate,an);

  /* Loop over midpoint */
  for(int i3 = 0; i3 < n3; ++i3) {
    /* Get the scan for the current midpoint */
    memcpy(scn,&allscn[i3*n1*n2],sizeof(float)*n1*n2);

    if(norm) normalize(n1,n2,scn);

    /* Transpose and reverse */
    for(int i2 = 0; i2 < n2; ++i2) {
      for(int i1 = 0; i1 < n1; ++i1) {
        wgt[i1*n2 + i2] = expf(-scn[i2*n1 + i1]);
      }
    }

    /* Do the picking */
    dp.find(i0, wgt);
    dp.traj(pck2 + i3*n1);

    /* Create ampl and pick for subsequent smoothing */
    for(int i1 = 0; i1 < n1; ++i1) {
      int i = i1 + i3*n1; // Index in the image space (nx,nz)
      float ct = pck2[i3*n1 + i1];
      pcko[i] = ct;
      int it = floorf(ct);
      ct -= it;
      if(it >= n2-1) {
        ampl[i] = scn[(n2-1)*n1 + i1];
      } else if(it < 0) {
        ampl[i] = scn[0*n1 + i1];
      } else {
        ampl[i] = scn[it*n1 + i1]*(1.-ct) + scn[(it+1)*n1 + i1]*ct;
      }
    }
  }

  /* Prepare for smoothing */
  float asum = 0.0;
  for(int i = 0; i < nm; ++i) {
    float a = ampl[i];
    asum += a*a;
  }
  asum = sqrtf(asum/nm);
  for(int i = 0; i < nm; ++i) {
    ampl[i] /= asum;
    pcko[i] = (o2+pcko[i]*d2-vel0)*ampl[i];
  }

  /* Free memory */
  delete[] scn; delete[] wgt;

}
