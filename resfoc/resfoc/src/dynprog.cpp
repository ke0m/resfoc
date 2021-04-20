#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "dynprog.h"

dynprog::dynprog(int nz, int nx, int gate, float an) {
  _n1 = nz; _n2 = nx; _gt = gate;

  /* Allocate memory */
  _next = new float[_n2]();
  _dist = new float[_n2]();
  _prev = new float[_n2]();
  _prob = new float[2*_gt-1]();
  _what = new float[_n1*_n2]();

  for(int i2 = 0; i2 < _n2; ++i2) {
    _dist[i2] = hypotf(i2,an);
  }

}

float dynprog::find_minimum(int ic, int nc, int jc, float c, float *pick) {

  float fm, f0, fp;

  if(ic == 0) {
    ic++;
    fm = c;
    f0 = _prob[ic  ];
    fp = _prob[ic+1];
  } else if (nc-1 == ic) {
    ic--;
    fm = _prob[ic-1];
    f0 = _prob[ic  ];
    fp=c;
  } else {
    fm = _prob[ic-1];
    f0 = c;
    fp = _prob[ic+1];
  }
  ic += jc;
  float a = fm + fp - 2.*f0;
  if (a <= 0.) { /* no minimum */
    if (fm < f0 && fm < fp) {
      *pick = ic-1;
      return fm;
    }
    if (fp < f0 && fp < fm) {
      *pick = ic+1;
      return fp;
    }
    *pick = ic;
    return f0;
  }
  float b = 0.5*(fm-fp);
  a = b/a;
  if (a > 1.) {
    *pick = ic+1;
    return fp;
  }
  if (a < -1.) {
    *pick = ic-1;
    return fm;
  }
  if (f0 < 0.5*b*a) {
    *pick = ic;
    return f0;
  }
  f0 -= 0.5*b*a;
  *pick=ic+a;

  return f0;
}

void dynprog::find(int i0, float *weight) {

  for (int i2=0; i2 < _n2; i2++) {
    float w = 0.5*(weight[1*_n2 + i2]+weight[0*_n2 + i0]);
    _prev[i2] = _dist[abs(i2-i0)]*w;
    _what[1*_n2 + i2] = i0;
  }

  for (int i1=2; i1 < _n1; i1++) {
    for (int i2=0; i2 < _n2; i2++) {
      float w = weight[i1*_n2 + i2];
      int ib = (i2-_gt < -1  ? -1 : i2-_gt);
      int ie = (i2+_gt < _n2 ? i2+_gt : _n2);
      float c = FLT_MAX;
      int ic = -1;
      for (int i=ib+1; i < ie; i++) {
        float w2 = 0.5*(w + weight[(i1-1)*_n2 + i]);
        float d = _dist[abs(i2-i)]*w2 + _prev[i];
        int it = i-ib-1;
        if (d < c) {
          c = d;
          ic = it;
        }
        _prob[it]=d;
      }

      _next[i2]= find_minimum(ic,ie-ib-1,ib+1,c,&_what[i1*_n2 + i2]);
    }
    for (int i2 = 0; i2 < _n2; i2++) {
      _prev[i2] = _next[i2];
    }
  }
}

void dynprog::traj(float *traj) {

  float c = FLT_MAX;
  float fc = 0;

  /* minimum at the bottom */
  for (int i2 = 0; i2 < _n2; i2++) {
    float d = _next[i2];
    if (d < c) {
      c = d;
      fc = i2;
    }
  }

  /* coming up */
  for (int i1 = _n1-1; i1 >= 0; i1--) {
    traj[i1]=fc;
    fc = interpolate(fc,i1);
  }
}

float dynprog::interpolate(float fc, int i1) {

  int ic = floorf(fc);
  fc -= ic;
  if (_n2-1 <= ic) return _what[i1*_n2 + _n2-1];
  if (0 > ic) return _what[i1*_n2 + 0];

  fc = _what[i1*_n2 + ic]*(1.-fc) + _what[i1*_n2 + ic+1]*fc;
  return fc;
}
