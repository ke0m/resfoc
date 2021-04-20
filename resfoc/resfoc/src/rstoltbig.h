/**
 * Residual stolt migration for large images
 * @author: Joseph Jennings
 * @version: 2020.04.03
 */

#ifndef RSTOLTBIG_H_
#define RSTOLTBIG_H_

class rstoltbig {
  public:
    rstoltbig(int nz, int nm, int nh, int nzp, int nmp, int nhp, int nro,
        float dz, float dm, float dh, float dro, float oro);
    void resmig(float *dat, float *img, int nthrd, bool verb);
    ~rstoltbig() {
      delete[] _ns; delete[] _signs; delete[] _s;
    }

  private:
    int _nz, _nm, _nh, _nzp, _nmp, _nhp, _nro;
    int _dim1, _n1, _n2;
    float _dz, _dm, _dh, _dro;
    float _oro;
    int *_ns, *_signs, *_s;
};

#endif /* RSTOLTBIG_H_ */
