/**
 * Inverse linear interpolation functions
 * (port of stretch.f90)
 * @author: Joseph Jennings
 * @version: 2019.12.11
 */

#ifndef STRETCH_H_
#define STRETCH_H_

class stretch {
public:
  /**
   * Stretch constructor
   * @param n1 length of the signal
   * @param o1 origin of signal
   * @param d1 sampling of signal
   * @param nd
   * @param epsilon regularization parameter
   */
  stretch(int n1, float o1, float d1, int nd, float epsilon=0.01);

  /**
   *  Defines the linear system to be inverted for interpolation
   *  from coord array, builds m, x, w, diag and offd
   * @param coord the input coordinates for the given trace
   * @param m the missing locations of the samples (output)
   * @param diag the diagonal of matrix (output)
   * @param offd the off-diagonal of the matrix (output)
   */
  void define(float *coord, bool *m, int *x, float *w, float *diag, float *offd);

  /**
   * Performs the inverse linear interpolation. Calls the function tridiag
   * @param coord input coordinates
   * @param ord input ordinates
   * @param mod
   */
  void solve(bool *m, int *x, float *w, float *diag, float *offd, float *ord, float *mod);

  /**
   * Applies the stretch operator. Calls first define and then solve
   * @param coord coordinates (input)
   * @param ord data (input)
   * @param mod output interpolated result (output)
   */
  void apply(float *coord, float *ord, float *mod);

  /**
   * Tridiagonal solver for symmetric systems
   * (Golub and van Loan, p. 157)
   * @param diag matrix diagonal
   * @param odiag off-diagonal
   * @param solution to A^-1 b
   */
  void tridiag(float *diag, float *odiag, float *b);

private:
  int _nt, _nz;
  float _t0, _dt, _eps;
};

#endif
