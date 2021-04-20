#include "depth2time.h"
#include "stretch.h"

void convert2time(int nh, int nm, int nz, float oz, float dz, int nt, float ot, float dt,
    float *vel, float *depth, float *time) {

  /* Create stretch object */
  stretch intrp = stretch(nt,ot,dt,nz,0.01);

  /* Create 1D temporary arrays */
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
    /* Perform the depth to time mapping */
    intrp.apply(depthm, depth + itr*nz, time + itr*nt);
  }

  /* Free memory */
  delete[] depthm;
}
