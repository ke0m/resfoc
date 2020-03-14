#include <cstring>
#include "depth2time.h"
#include "stretch.h"

void convert2time(int nh, int nm, int nz, float oz, float dz, int nt, float ot, float dt,
    float *vel, float *depth, float *time) {

  /* Create stretch object */
  stretch intrp = stretch(nt,ot,dt,nz,0.01);

  /* Create 1D temporary arrays */
  float *dpthtr = new float[nz]();
  float *timetr = new float[nt]();
  float *depthm = new float[nz]();

  int ntr = nm*nh;

  /* Loop over all traces */
  for(int itr = 0; itr < ntr; ++itr) {
    float z = 0;
    /* Compute the map for the current trace */
    for (int iz = 0; iz < nz; ++iz) {
      if(iz != 0) {
        z += 1.0/vel[itr*nz + iz-1];
      }
      depthm[iz] = 2.0*dz*z;
    }
    /* Get one trace */
    memcpy(dpthtr,&depth[itr*nz],sizeof(float)*nz);
    /* Initialize output trace to 0 */
    memset(timetr, 0, sizeof(float)*nt);
    /* Perform the depth to time mapping */
    intrp.apply(depthm, dpthtr, timetr);
    /* Copy it to the output */
    memcpy(&time[itr*nt],timetr,sizeof(float)*nt);
  }

  /* Free memory */
  delete[] dpthtr; delete[] timetr; delete[] depthm;
}