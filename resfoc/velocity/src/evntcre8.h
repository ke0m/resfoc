/**
 * Creates geological events as part of a geological
 * model building code
 * @author: Joseph Jennings
 * @version: 2020.02.03
 */
#ifndef EVNTCRE8_H_
#define EVNTCRE8_H_

#include <vector>

class evntcre8 {
  public:
    evntcre8(int nx, int ny, float dx, float dy, float dz);
    /* Main worker functions */
    void expand(int itop, int ibot, int nzin, int *lyrin, float *velin, int nzot, int *lyrot, float *velot);
    void deposit(float vel,
        float band1, float band2, float band3,
        float layerT, float layer_rand, float dev_layer, float dev_pos,
        int nzot, int *lyrot, float *velot);
    void fault(int nz, int *lyrin, float *velin, float *lblin, float azim, float begx, float begy, float begz, float dz, float daz,
        float thetashift, float perpdie, float distdie, float thetadie, float dir, float scalethrw,
        int *lyrot, float *velot, float *olblot, float *nlblot);
    void fault_shifts2d(int nz, float *lblin, float azim, float begx, float begz, float dz, float daz,
        float thetashift, float distdie, float thetadie, float scalethrw,
        float *olblot, float *nlblot, float *shiftx, float *shiftz);
    void squish(int nz, int *lyrin, float *velin, float *shftin, int mode,
        float azim, float maxshift, float lambda, float rinline, float rxline,
        int nzot, int *lyrot, float *velot);
    void squish_shifts(int nz, float *shftin, int mode,
        float azim, float lambda, float rinline, float rxline,
        float *shfty, float *shftx, float *shftz);
    void fill_top_bottom(int nz, float maxshift, float basevel, int *lyrot, float *velot);
    /* Deposit helper functions */
    int find_max_deposit(int n1, int n2, int n3, int *lyrin);
    void fill_random(int n1, int n2, float *velot);
    void fill_random(int n1, int n2, int n3, float *velot);
    void bandpass(float f1, float f2, float f3, int n1, int n2, int n3, float *velin);
    std::vector<float> find_lim1(int n, float f);
    std::vector<float> find_lim2(int n, float f);
    float find_max(int n1, int n2, int n3, float *arr);
    float find_min(int n1, int n2, int n3, float *arr);
    float find_absmax(int n1, int n2, int n3, float *arr);
    void scale(int n1, int n2, int n3, float *arr, float sc);
    void scale(int n1, int n2, float *arr, float sc);
    void add(int n1, int n2, float *arr, float sc);
    void norm(int n1, int n2, int n3, float *arr, float sc);
    /* Fault helper functions */
    void zder(int nz, float *lblin, float *lblot);
    void laplacian(int nz, float *lblin, float *lblot);
    /* Fold helper functions */
    void calcshift_fastaxis(int nn, float d2, float *shift, float *dist);
    void rotate_array(float cs, float sn, int nn, float *shift, int n2, float d2, int n3, float d3, float *shiftrot);
    void smooth(int n1, int n2, float *arr, int rect1, int rect2);
    void recForward (std::vector<float> &vec1, std::vector<float> &vec2, const int len);
    void recBackward(std::vector<float> &vec1, std::vector<float> &vec2, const int len);
    /* Other functions */
    void calcref(int nz, float *vel, float *ref);
    void calcref2d(int nx, int nz, float *vel, float *ref);

  private:
    int _n2, _n3;
    float _d2, _d3, _d1;
};

#endif /* EVNTCRE8_H_ */
