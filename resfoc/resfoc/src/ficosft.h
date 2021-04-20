/**
 * Applies the forward and inverse cosine Fourier Transforms
 * A port of Mcosft.c from Madagascar written by Sergey Fomel
 * @author: Joseph Jennings
 */
#ifndef FICOSFT_H_
#define FICOSFT_H_

#include "cosft.h"

void fwdcosft(int dim1, int n1, int n2, int *n, int *sign, int *s, float *data, bool verb);
void invcosft(int dim1, int n1, int n2, int *n, int *sign, int *s, float *data, bool verb);
int first_index(int i, int j, int dim, const int *n, const int *s);

#endif /* FICOSFT_H_ */
