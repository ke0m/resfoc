#include "ficosft.h"
#include "progressbar/progressbar.h"

void fwdcosft(int dim1, int n1, int n2, int *n, int *sign, int *s, float *data, bool verb) {
  int ctr = 0;
  for(int i2 = 0; i2 < n2; i2++) {
    for(int i = 0; i <= dim1; ++i) {
      if(verb) printprogress("npass:",ctr,(dim1+1)*n2);
      if(!sign[i]) continue;
      /* Create cosine transform object */
      cosft cft = cosft(n[i]);
      for(int j = 0; j < n1/n[i]; ++j) {
        int i0 = first_index(i,j,dim1+1,n,s);
        cft.fwd(data + i2*n1, i0, s[i]);
      }
      /* Progress counter */
      ctr++;
    }
  }
}

void invcosft(int dim1, int n1, int n2, int *n, int *sign, int *s, float *data, bool verb) {
  int ctr = 0;
  for(int i2 = 0; i2 < n2; i2++) {
    for(int i = 0; i <= dim1; ++i) {
      if(verb) printprogress("npass:",ctr,(dim1+1)*n2);
      if(!sign[i]) continue;
      /* Create cosine transform object */
      cosft cft = cosft(n[i]);
      for(int j = 0; j < n1/n[i]; ++j) {
        int i0 = first_index(i,j,dim1+1,n,s);
        cft.inv(data + i2*n1, i0, s[i]);
      }
      /* Progress counter */
      ctr++;
    }
  }
}

int first_index (int i /* dimension [0...dim-1] */,
    int j              /* line coordinate */,
    int dim            /* number of dimensions */,
    const int *n       /* box size [dim] */,
    const int *s       /* step [dim] */)
/*< Find first index for multidimensional transforms >*/
{
  int i0, n123, ii;
  int k;

  n123 = 1;
  i0 = 0;
  for (k=0; k < dim; k++) {
    if (k == i) continue;
    ii = (j/n123)%n[k]; /* to cartesian */
    n123 *= n[k];
    i0 += ii*s[k];      /* back to line */
  }

  return i0;
}
