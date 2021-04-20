/**
 * Functions for converting depth images to time images
 * @author: Joseph Jennings
 * @version: 2020.02.25
 */

#ifndef DEPTH2TIME_H_
#define DEPTH2TIME_H_

void convert2time(int nh, int nm, int nz, float oz, float dz, int nt, float ot, float dt,
    float *vel, float *depth, float *time);

#endif /* DEPTH2TIME_H_ */
