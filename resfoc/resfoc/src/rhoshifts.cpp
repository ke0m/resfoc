#include "rhoshifts.h"

void rhoshifts(int nro, int nx, int nz, float dro, float *rho, float *coords) {

  for(int iro = 0; iro < nro; ++iro) {
    for(int ix = 0; ix < nx; ++ix) {
      for(int iz = 0; iz < nz; ++iz) {
        int idx = iro*nx*nz + ix*nz + iz;
        coords[0*nro*nx*nz + idx] = (rho[ix*nz + iz]-1.0)/dro + iro;
        coords[1*nro*nx*nz + idx] = ix;
        coords[2*nro*nx*nz + idx] = iz;
      }
    }
  }
}
