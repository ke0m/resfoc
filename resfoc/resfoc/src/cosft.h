/**
 * Performs the cosine fourier transform (real to real)
 * A port of cosft.c from Madagascar written by Sergey Fomel
 * @author: Joseph Jennings
 * @version: 2020.04.02
 */
#ifndef COSFT_H_
#define COSFT_H_

#include <complex>
#include <fftw3.h>

class cosft {
  public:
  cosft();
  cosft(int n1_in);
  void fwd(float *q, int o1, int d1);
  void inv(float *q, int o1, int d1);
  ~cosft() {
    delete [] _pp; delete[] _p;
    fftwf_destroy_plan(_fplan); fftwf_destroy_plan(_iplan);
  }

  private:
  int _nt, _nw, _n1;
  float *_p;
  std::complex<float> *_pp;
  fftwf_plan _fplan, _iplan;
};

#endif /* COSFT_H_ */
