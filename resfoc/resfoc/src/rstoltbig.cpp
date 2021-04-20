#include <omp.h>
#include <math.h>
#include <cstring>
#include "stretch.h"
#include "ficosft.h"
#include "rstoltbig.h"
#include "progressbar/progressbar.h"

rstoltbig::rstoltbig(int nz, int nm, int nh, int nzp, int nmp, int nhp, int nro,
    float dz, float dm, float dh, float dro, float oro) {
  /* Sizes */
  _nz  = nz;  _nm  = nm;  _nh  = nh;
  _nzp = nzp; _nmp = nmp; _nhp = nhp; _nro = 2*nro-1;
  /* Samplings */
  _dz = M_PI*dz; _dm = M_PI*dm; _dh = M_PI*dh; _dro = dro;
  /* Origin */
  _oro = oro - (nro-1)*dro;
  /* Set up inverse cosine transform inputs */
  _ns    = new int[3]();  _signs = new int[3](); _s = new int[3]();
  _dim1 = 2; _n1 = _nzp*_nmp*_nhp; _n2 = 1;
	if(_nhp > 1) { // Prestack
    _ns[0] = _nzp; _signs[0] = 1; _s[0] = 1;
    _ns[1] = _nmp; _signs[1] = 1; _s[1] = _nzp;
    _ns[2] = _nhp; _signs[2] = 1; _s[2] = _nzp*_nmp;
	} else {      // Poststack
    _ns[0] = _nzp; _signs[0] = 1; _s[0] = 1;
    _ns[1] = _nmp; _signs[1] = 1; _s[1] = _nzp;
    _ns[2] = _nhp; _signs[2] = 0; _s[2] = 0;
	}
}

void rstoltbig::resmig(float *dat, float *img, int nthrd, bool verb) {

  /* Initialize stretch */
  stretch intrp = stretch(_nzp,0.0,_dz,_nzp,0.01);

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
		//TODO: set two types of verbosity. depending on the number of threads used
    if(firstiter && verb) ridx[tidx] = iro;
		if(verb) {
			if(nthrd >= _nro) {
				printprogress_omp("resmig:", 0, 2, tidx);
			} else {
				printprogress_omp("nrho:", iro - ridx[tidx], csize, tidx);
			}
		}
    /* Temporary arrays */
    float *iimg = new float[_nzp*_nmp*_nhp]();
    float *str  = new float[_nzp]();
    /* Compute rho */
    float vov = _oro + iro*_dro;
    /* Loop over sub-surface offset */
    for(int ih = 0; ih < _nhp; ++ih) {
      float kh = ih*_dh;
      /* Loop over midpoint (image point) */
      for(int im = 0; im < _nmp; ++im) {
        float km = im*_dm;
        /* Create the mapping z -> z' (loop from iz=1 to avoid kz=0) */
        for(int iz = 1; iz < _nzp; ++iz) {
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
        int idx = ih*_nzp*_nmp + im*_nzp;
        intrp.apply(str, dat + idx, iimg + idx);
      }
    }
    /* Inverse cosine transform */
    if(verb && nthrd >= _nro) printprogress_omp("cosftr:", 1, 2, tidx);
    invcosft(_dim1, _n1, _n2, _ns, _signs, _s, iimg, false);
    /* Copy to output */
    for(int ih = 0; ih < _nh; ++ih) {
      for(int im = 0; im < _nm; ++im) {
        memcpy(&img[iro*_nz*_nm*_nh + ih*_nz*_nm + im*_nz],&iimg[ih*_nzp*_nmp + im*_nzp],sizeof(float)*_nz);
      }
    }
    /* Parallel printing */
    firstiter = false;
    /* Free memory */
    delete[] str; delete[] iimg;
  }
  if(verb && nthrd >= _nro) printprogress_omp("finish!", 2, 2, 0);
  /* Parallel printing */
  if(verb) printf("\n");
  delete[] ridx;
}
