/**
 * Dynamical programming for automatically
 * picking semblance panels.
 * A port of the dynprog.c code in Madagascar
 * written by Sergey Fomel
 *
 * @author: Joseph Jennings
 * @version: 2020.06.16
 */

#ifndef DYNPROG_H_
#define DYNPROG_H_

class dynprog {
  public:
    dynprog();
    dynprog(int nz, int nx, int gate, float an);
    void find(int i0, float *weight);
    void traj(float *traj);
    float find_minimum(int ic, int nc, int jc, float c, float *pick);
    float interpolate(float fc, int i1);
    ~dynprog(){
      delete[] _prev; delete[] _next;
      delete[] _dist; delete[] _prob;
      delete[] _what;
    }

  private:
    int _n1, _n2, _gt;
    float *_prev, *_next, *_dist, *_prob, *_what;
};

#endif /* DYNPROG_H_ */
