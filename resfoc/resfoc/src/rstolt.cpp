#include <omp.h>
#include <math.h>
#include "stretch.h"
#include "rstolt.h"
#include "progressbar/progressbar.h"

rstolt::rstolt(int nz, int nm, int nh, int nro, float dz, float dm, float dh, float dro, float oro) {
  /* Sizes */
  _nz = nz; _nm = nm; _nh = nh; _nro = 2*nro-1;
  /* Samplings */
  _dz = M_PI*dz; _dm = M_PI*dm; _dh = M_PI*dh; _dro = dro;
  /* Origin */
  _oro = oro - (nro-1)*dro;
}

void rstolt::resmig(float *dat, float *img, int nthrd, bool verb) {

  /* Initialize stretch */
  stretch intrp = stretch(_nz,0.0,_dz,_nz,0.01);

  /* Set up printing if verbosity is desired */
  int *ridx = new int[nthrd]();
  int csize = (int)_nro/nthrd;
  if(_nro%nthrd != 0) csize += 1;
  bool firstiter = true;

  omp_set_num_threads(nthrd);
  /* Loop over rho */
#pragma omp parallel for default(shared)
  for(int iro = 0; iro < _nro; ++iro) {
    /* Set up the parallel printing */
    int tidx = omp_get_thread_num();
    if(firstiter && verb) ridx[tidx] = iro;
    if(verb) printprogress_omp("nrho:", iro - ridx[tidx], csize, tidx);
    /* Temporary arrays */
    float *str = new float[_nz]();
    /* Compute rho */
    float vov = _oro + iro*_dro;
    /* Loop over sub-surface offset */
    for(int ih = 0; ih < _nh; ++ih) {
      float kh = ih*_dh;
      /* Loop over midpoint (image point) */
      for(int im = 0; im < _nm; ++im) {
        float km = im*_dm;
        /* Create the mapping z -> z' (loop from iz=1 to avoid kz=0) */
        for(int iz = 1; iz < _nz; ++iz) {
          float kz = iz*_dz;
          /* Dispersion relation */
          float kzh = kz*kz + kh*kh;
          float kzm = kz*kz + km*km;
          float zzs = (vov*vov) * (kzh*kzm) - (kz*kz) * ( (km-kh)*(km-kh) );
          float zzg = (vov*vov) * (kzh*kzm) - (kz*kz) * ( (km+kh)*(km+kh) );
          if(zzs > 0 && zzg > 0) {
            str[iz] = 0.5/kz * ( sqrt(zzs) + sqrt(zzg) );
          } else { /* Evanescent */
            str[iz] = -2.0*_dz;
          }
        }
        /* Do the migration for the mapping */
        long long datidx = ih*_nz*_nm + im*_nz;
        long long imgidx = iro*_nz*_nm*_nh + ih*_nz*_nm + im*_nz;
        intrp.apply(str, dat + datidx, img + imgidx);
      }
    }
    /* Parallel printing */
    firstiter = false;
    /* Free memory */
    delete[] str;
  }
  /* Parallel printing */
  if(verb) printf("\n");
  delete[] ridx;
}
