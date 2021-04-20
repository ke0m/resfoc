/**
 * Automatic semblance picker
 * A port of sfpick from Madagascar written by Sergey Fomel
 *
 * @author: Joseph Jennings
 * @version: 2020.06.20
 */

#ifndef PICKSCAN_H_
#define PICKSCAN_H_

void normalize(int n1, int n2, float *scn);

void pickscan(float an, int gate, bool norm, float vel0, float o2, float d2,
              int n1, int n2, int n3, float *allscn, float *pck2, float *ampl, float *pcko);


#endif /* PICKSCAN_H_ */
