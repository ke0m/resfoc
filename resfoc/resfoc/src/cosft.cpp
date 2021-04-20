#include <fftw3.h>
#include "cosft.h"
#include "kiss_fft.h"

cosft::cosft(int n1_in) {
  /* Get sizes */
  _n1 = n1_in;
  _nt = 2*kiss_fft_next_fast_size(_n1-1);
  _nw = _nt/2 + 1;
  /* Temporary arrays */
  _p = new float[_nt]();
  _pp = new std::complex<float>[_nw]();
  /* Forward and inverse FFT objects */
  _fplan = fftwf_plan_dft_r2c_1d(_nt,
                                 _p,reinterpret_cast<fftwf_complex*>(_pp),
                                 FFTW_ESTIMATE);
  _iplan = fftwf_plan_dft_c2r_1d(_nt,
                                 reinterpret_cast<fftwf_complex*>(_pp),_p,
                                 FFTW_ESTIMATE);
}

void cosft::fwd(float *q, int o1, int d1) {
  /* Copy data */
  for(int i = 0; i < _n1; ++i) {
    _p[i] = q[o1+i*d1];
  }
  /* Pad */
  for(int i = _n1; i < _nw; ++i) {
    _p[i] = 0.0;
  }
  for(int i = _nw; i < _nt; ++i) {
    _p[i] = _p[_nt-i];
  }
  /* Forward transform */
  fftwf_execute(_fplan);
  /* Copy real part */
  for(int i = 0; i < _n1; ++i) {
    q[o1+i*d1] = std::real(_pp[i]);
  }
}

void cosft::inv(float *q, int o1, int d1) {
  /* Copy data */
  for(int i = 0; i < _n1; ++i) {
    _pp[i] = std::complex<float>(q[o1+i*d1],0.0);
  }
  /* Pad */
  for(int i = _n1; i < _nw; ++i) {
    _pp[i] = std::complex<float>(0.0,0.0);
  }
  /* Inverse transform */
  fftwf_execute(_iplan);
  /* Copy to output */
  for(int i = 0; i < _n1; ++i) {
    q[o1+i*d1] = _p[i]/_nt;
  }
}
