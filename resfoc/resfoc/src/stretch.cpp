
#include <math.h>
#include <stdio.h>
#include "stretch.h"

stretch::stretch(int n1, float o1, float d1, int nd, float epsilon) {
  _nt = n1; _t0 = o1; _dt = d1;  //Input signal axis
  _nz = nd; // Output length
  _eps = epsilon;
}

void stretch::define(float *coord, bool *m, int *x, float *w, float *diag, float *offd) {

  /* Initialize elements of diagonal and off-diagonal */
  for(int it = 0; it < _nt; ++it) {
    diag[it] = 2.0*_eps; offd[it] = -_eps; // Regularization
  }

  for(int id = 0; id < _nz; ++id) {
    /* Standard interpolation step */
    float rx = (coord[id] - _t0)/_dt;
    int ix = floor(rx);
    rx -= ix;
    if(ix < 0 || ix > _nt -2) {
      m[id] = true; continue;
    }
    x[id] = ix; m[id] = false; w[id] = rx;
    int i1 = ix; int i2 = i1 + 1;
    /* Interpolation weights */
    float w2 = rx; float w1 = 1 - w2;
    /* Compute elements of matrices */
    diag[i1] += w1*w1;
    diag[i2] += w2*w2;
    offd[i1] += w1*w2;
  }
}

void stretch::solve(bool *m, int *x, float *w, float *diag, float *offd, float *ord, float *mod) {

  /* Build right hand side */
  for(int id = 0; id < _nz; ++id) {
    if(m[id]) continue;
    int i1 = x[id]; int i2 = i1 + 1;
    float w2 = w[id]; float w1 = 1 - w2;
    mod[i1] += w1*ord[id];
    mod[i2] += w2*ord[id];
  }
  /* Invert system */
  tridiag(diag,offd,mod);
}

void stretch::apply(float *coord, float *ord, float *mod) {

  /* Allocate memory */
  bool *m     = new bool[_nz]();
  int  *x     = new int[_nz]();
  float *w    = new float[_nz]();
  float *diag = new float[_nt]();
  float *offd = new float[_nt]();

  /* Create linear system */
  define(coord,m,x,w,diag,offd);
  /* Solve linear system */
  solve(m,x,w,diag,offd,ord,mod);

  /* Free memory */
  delete[] x; delete[] m;
  delete[] w; delete[] diag; delete[] offd;

}

void stretch::tridiag(float *diag, float *offd, float *b) {

  float *d = new float[_nt]();
  float *o = new float[_nt]();

  d[0] = diag[0];
  for(int k = 1; k < _nt; ++k) {
    float t = offd[k-1]; o[k-1] = t/diag[k-1]; d[k] = diag[k] - t*o[k-1];
  }
  for(int k = 1; k < _nt; ++k) {
    b[k] = b[k] - o[k-1]*b[k-1];
  }
  b[_nt-1] = b[_nt-1]/d[_nt-1];
  for(int k = _nt-2; k >= 0; --k) {
    b[k] = b[k]/d[k] - o[k]*b[k+1];
  }

  delete[] d; delete[] o;

}
