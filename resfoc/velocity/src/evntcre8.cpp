#include <math.h>
#include <algorithm>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>
#include <fftw3.h>
#include "evntcre8.h"

evntcre8::evntcre8(int nx, int ny, float dx, float dy, float dz) {
  _n2 = nx; _n3 = ny;
  _d2 = dx; _d3 = dy; _d1 = dz;
}

void evntcre8::expand(int itop, int ibot, int nzin, int *lyrin, float *velin, int nzot, int *lyrot, float *velot) {

  //TODO: should check nzot with nzin and itop and ibot
  /* Assign the expanded model indices in lyrot */
  tbb::parallel_for(tbb::blocked_range<int>(0, _n3),
      [&](const tbb::blocked_range<int>& r) {
    for (int i3 = r.begin(); i3 != r.end(); ++i3) {
      for (int i2 = 0; i2 < _n2; ++i2) {
        for (int i1 = 0; i1 < itop; ++i1) {
          lyrot[i3*nzot*_n2 + i2*nzot + i1] = -1;
          velot[i3*nzot*_n2 + i2*nzot + i1] = -1.0;
        }
        for (int i1 = 0; i1 < nzin; ++i1) {
          lyrot[i3*nzot*_n2 + i2*nzot + i1 + itop] = lyrin[i3*nzin*_n2 + i2*nzin + i1];
          velot[i3*nzot*_n2 + i2*nzot + i1 + itop] = velin[i3*nzin*_n2 + i2*nzin + i1];
        }
        for (int i1 = 0; i1 < ibot; ++i1) {
          lyrot[i3*nzot*_n2 + i2*nzot + i1 + nzin + itop] = -1;
          velot[i3*nzot*_n2 + i2*nzot + i1 + nzin + itop] = -1.0;
        }
      }
    }
  });

}

//TODO:  probably need to pass in the current event number
void evntcre8::deposit(float vel,
    float band1, float band2, float band3,
    float layerT, float layer_rand, float dev_layer, float dev_pos,
    int nzot, int *lyrot, float *velot) {

  /* Find the thickness of the deposit based on the output layer and layer model*/
  int n1use = std::max(16, find_max_deposit(nzot,_n2,_n3,lyrot));

  /* Build a 1D function for defining the fine layering */
  int iold = 0;
  float vu = (1. + ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) - .5) * dev_layer);
  std::vector<float> layv(nzot, 0.);

  float luse = layerT * (1 + (.5 * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX))) * layer_rand);

  for (int i1 = 0; i1 < nzot; i1++) {
    int ii = (i1 - iold) * _d1 / luse;
    if (ii != iold) {
      vu = (1. + ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) - .5) * dev_layer);
    }
    iold = ii;
    luse = layerT * (1 + (.5 * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX))) * layer_rand);
    layv[i1] = vu;
  }

  /* Start by filling with bandpassed random numbers */
  float *vtmp = new float[n1use*_n2*_n3]();
  if(dev_pos != 0.0) {
    fill_random(n1use, _n2, _n3, vtmp);
    bandpass(band1, band2, band3, n1use, _n2, _n3, vtmp);
    norm(n1use,_n2,_n3,vtmp,1);
  }

  /* First compute the entire temporary array based on the max computed height */
  tbb::parallel_for(tbb::blocked_range<size_t>(0, _n3),
      [&](const tbb::blocked_range<size_t> &r) {
    for (size_t i3 = r.begin(); i3 != r.end(); ++i3) {
      for (int i2 = 0; i2 < _n2; i2++) {
        for (int i1 = 0; i1 < n1use; i1++) {
          vtmp[i3*_n2*n1use + i2*n1use + i1] = vel * layv[i1] * (1 + vtmp[i3*_n2*n1use + i2*n1use + i1] * dev_pos);
        }
      }
    }
  });

  /* Now fill in the layer based on the actual layer position (layer field) */
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, _n3),
      [&](const tbb::blocked_range<size_t> &r) {
    for (size_t i3 = r.begin(); i3 != r.end(); ++i3) {
      for (int i2 = 0; i2 < _n2; ++i2) {
        int i1 = 0;
        while (i1 < nzot - 1 && lyrot[i3*_n2*nzot + i2*nzot + i1] == -1) i1++;
        for (int ia = 0; ia < i1; ++ia) {
          velot[i3*_n2*nzot + i2*nzot + ia] = vtmp[i3*_n2*n1use + i2*n1use + ia];
          lyrot[i3*_n2*nzot + i2*nzot + ia] = 1;
        }
      }
    }
  });

  /* Clean up */
  delete[] vtmp;
}

void evntcre8::fault(int nz, int *lyrin, float *velin, float *lblin, float azim, float begx, float begy, float begz, float dz, float daz,
    float thetashift, float perpdie, float distdie, float thetadie, float dir, float scalethrw,
    int *lyrot, float *velot, float *olblot, float *nlblot) {

  float dst1 = nz*_d1;
  float dst2 = _n2*_d2;
  float dst3 = _n3*_d3;

  /* Location of the beginning of the tear */
  float zbeg = dst1 * begz;
  float ybeg = dst3 * begy;
  float xbeg = dst2 * begx;

  // Die off distance from the tear perpendicular to the tear and radius direction
  distdie *= dst3;
  perpdie *= dst3;

  // Definition of pi
  float pi = atan(1.) * 4;

  float theta0 = atan2f(dz, daz);
  theta0 = theta0 + 2 * pi;
  theta0 = theta0 * 180. / pi;

  /* Azimuth normal to fault */
  if (azim < 0.) azim += 360.;  // Put azimuth in the 0-360
  azim *= pi / 180.;            // convert to radians
  float naz1   =  cosf(azim);   // Rotation matrix 1
  float naz2   = -sinf(azim);   // Rotation matrix 2
  float nperp1 =  sinf(azim);   // Rotation matrix 3
  float nperp2 =  cosf(azim);   // Rotation matrix 4

  /* Cylindrical coordinate 0,0,0 is at */
  float zcenter = zbeg - dz;
  float xcenter = xbeg - daz * cosf(azim);
  float ycenter = ybeg - daz * sinf(azim);

  float fullRadius = sqrtf(dz * dz + daz * daz);

  /* Create shift matrices */
  float *shiftx = new float[nz*_n2*_n3]();
  float *shifty = new float[nz*_n2*_n3]();
  float *shiftz = new float[nz*_n2*_n3]();

  /* Compute shifts */
  for (int i3 = 0; i3 < _n3; i3++) {
    // Y component of distance from center
    float p3 = _d3 * i3 - ycenter;
    for (int i2 = 0; i2 < _n2; i2++) {
      // X component of distance from center
      float p2 = _d2 * i2 - xcenter;

      // Rotate coordinate x,y to along azimuth and perpendicular
      // Applies rotation here to rotate into the desired azimuth
      float azP   =  naz1   * p2 - naz2   * p3;
      float perpP = -nperp1 * p2 + nperp2 * p3;

      // Ratio die off along azimuth in 2D
      float ratioAz = fabsf(fullRadius - azP) / distdie;
      // Ratio die off along perp
      float ratioPerp = perpP / perpdie;

      float scalePerp = 1. - fabsf(ratioPerp);

      // If we are less than die off in perp and along azimuth
      for (int i1 = 0; i1 < nz; i1++) {
        shiftz[i3*_n2*nz + i2*nz + i1] = 0;
        shiftx[i3*_n2*nz + i2*nz + i1] = 0;
        shifty[i3*_n2*nz + i2*nz + i1] = 0;
      }

      if (fabsf(ratioPerp) < 1.) {
        for (int i1 = 0; i1 < nz; i1++) {
          // Z component of distance from center
          float p1 = _d1 * i1 - zcenter;

          // Theta of our current point
          float thetaOld = atan2f(p1, azP) * 180. / pi;
          thetaOld += 360;
          float thetaCompare = atanf(p1 / azP) * 180. / pi + 360.;

          // True radius of current point
          float radius = sqrtf(azP * azP + p1 * p1);

          ratioAz = fabsf(fullRadius - radius) / distdie;
          float ratioTheta = fabsf(thetaCompare - theta0) / thetadie;

          // Compute distance from xbeg, ybeg and zbeg
          float diffx = xbeg - (p2 + xcenter);
          float diffy = ybeg - (p3 + ycenter);
          float diffz = zbeg - (p1 + zcenter);
          float distbeg = sqrtf(diffx*diffx + diffy*diffy + diffz*diffz);

          if (ratioAz < 1. && ratioTheta < 1. && distbeg < 8000) {
            float scaleAz    = 1. - ratioAz;
            float scaleTheta = 1. - ratioTheta;

            // Shift in theta
            float shiftTheta = thetashift * scaleAz * scaleTheta * scalePerp;

            // New theta location
            float thetaNew = thetaOld + shiftTheta;
            if (dir < 0. || radius > fullRadius)
              thetaNew = thetaOld - shiftTheta;


            // Convert to polar coordinates
            float dPR  = radius * cosf(thetaNew * pi / 180.);
            float newZ = radius * sinf(thetaNew * pi / 180.) + zcenter;

            // Now rotate back to standard coordinate system
            float newX = naz1   * dPR + naz2   * perpP + xcenter;
            float newY = nperp1 * dPR + nperp2 * perpP + ycenter;

            // Compute shifts to be applied
            shiftz[i3*nz*_n2 + i2*nz + i1] = (newZ - (_d1 * i1))/scalethrw;
            shiftx[i3*nz*_n2 + i2*nz + i1] = (newX - (_d2 * i2))/scalethrw;
            shifty[i3*nz*_n2 + i2*nz + i1] = (newY - (_d3 * i3))/scalethrw;

            // Save new label out
            nlblot[i3*nz*_n2 + i2*nz + i1] = shiftz[i3*nz*_n2 + i2*nz + i1];
          }
        }
        bool found = false;
        int i1 = 0;
        while (i1 < nz - 1 && !found) {
          if (fabs(shiftz[i3*nz*_n2 + i2*nz + i1]) > 0.)
            found = true;
          else
            i1++;
        }
        if (found) {
          for (int i = 0; i <= i1; i++) {
            shiftz[i3*nz*_n2 + i2*nz + i] = shiftz[i3*nz*_n2 + i2*nz + i1];
          }
        }
      }
    }
  }

  /* Compute output layer and velocity arrays */
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, _n3),
      [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i3 = r.begin(); i3 != r.end(); ++i3) {
      for (int i2 = 0; i2 < _n2; i2++) {
        for (int i1 = 0; i1 < nz; i1++) {
          int l1 = std::max(0, (int)(i1 - shiftz[i3*nz*_n2 + i2*nz + i1] / _d1 + .5));
          int l2 = std::max(0, (int)(i2 - shiftx[i3*nz*_n2 + i2*nz + i1] / _d2 + .5));
          int l3 = std::max(0, (int)(i3 - shifty[i3*nz*_n2 + i2*nz + i1] / _d3 + .5));
          if (l1 >=  nz) l1 =  nz - 1;
          if (l2 >= _n2) l2 = _n2 - 1;
          if (l3 >= _n3) l3 = _n3 - 1;
          if (l1 >= 0) {
            lyrot [i3*nz*_n2 + i2*nz + i1] = lyrin[l3*nz*_n2 + l2*nz + l1];
            velot [i3*nz*_n2 + i2*nz + i1] = velin[l3*nz*_n2 + l2*nz + l1];
            olblot[i3*nz*_n2 + i2*nz + i1] = lblin[l3*nz*_n2 + l2*nz + l1];
          }
          else {
            lyrot [i3*nz*_n2 + i2*nz + i1] = -1;
            velot [i3*nz*_n2 + i2*nz + i1] =  0;
            olblot[i3*nz*_n2 + i2*nz + i1] =  0;
          }
        }
      }
    }
  });

  /* Free memory */
  delete[] shiftx; delete[] shifty; delete[] shiftz;

}

void evntcre8::fault_shifts2d(int nz, float *lblin, float azim, float begx, float begz, float dz, float daz,
    float thetashift, float distdie, float thetadie, float scalethrw,
    float *olblot, float *nlblot, float *oshiftx, float *oshiftz) {

  float dst1 = nz*_d1;
  float dst2 = _n2*_d2;
  float dst3 = _n3*_d3;

  /* Location of the beginning of the tear */
  float zbeg = dst1 * begz;
  float xbeg = dst2 * begx;

  // Die off distance from the tear perpendicular to the tear and radius direction
  distdie *= dst3;

  // Definition of pi
  float pi = atan(1.) * 4;

  /* Angle of fault */
  float theta0 = atan2f(dz, daz);
  theta0 = theta0 + 2 * pi;
  theta0 = theta0 * 180. / pi;

  /* Azimuth normal to fault */
  if (azim < 0.) azim += 360.;  // Put azimuth in the 0-360
  azim *= pi / 180.;            // convert to radians
  float sign   =  cosf(azim);   // Make either 0 or 180

  /* Cylindrical coordinate 0,0,0 is at */
  float zcenter = zbeg - dz;
  float xcenter = xbeg - daz * sign;

  /* Distance from fault to center */
  float fullRadius = sqrtf(dz * dz + daz * daz);

  /* Internal shift arrays */
  float *shiftx = new float[_n2*nz];
  float *shiftz = new float[_n2*nz];

  /* Compute shifts */
  for (int i2 = 0; i2 < _n2; i2++) {
    // X component of distance from center
    float p2 = _d2 * i2 - xcenter;
    // Rotate 180 if desired
    p2 *= sign;

    for (int i1 = 0; i1 < nz; i1++) {
      // Z component of distance from center
      float p1 = _d1 * i1 - zcenter;

      // Save output shift values
      oshiftx[i2*nz + i1] = i2;
      oshiftz[i2*nz + i1] = i1;

      // Compute the angle from vertical for the current point
      float thetaOld = atan2f(p1, p2) * 180. / pi + 360;

      // Radius of current point
      float radius = sqrtf(p2 * p2 + p1 * p1);

      // Save fault trajectory for label
      float rdiff = fabsf(radius - fullRadius);
      if(rdiff < 20) {
        nlblot[i2*nz + i1] = 1.0;
      }

      // Check if we are in the region for faulting
      // Criteria for radius and angle
      float ratioAz    = rdiff / distdie;
      float ratioTheta = fabsf(thetaOld - theta0) / thetadie;

      if (ratioAz < 1. && ratioTheta < 1.) {
        // Once we are in range compute the shift in angle
        float scaleAz    = 1. - ratioAz;
        float scaleTheta = 1. - ratioTheta;

        // Shift in theta
        float shiftTheta = thetashift * scaleAz * scaleTheta;

        // New theta location
        float thetaNew = thetaOld + shiftTheta;
        // If outside the circle, flip the sign of the shift
        if (radius > fullRadius)
          thetaNew = thetaOld - shiftTheta;

        // Convert to polar coordinates
        float newX = radius * cosf(thetaNew * pi / 180.) * sign + xcenter;
        float newZ = radius * sinf(thetaNew * pi / 180.)        + zcenter;

        // Compute shifts to be applied
        shiftz[i2*nz + i1] = (newZ - (_d1 * i1))/scalethrw;
        shiftx[i2*nz + i1] = (newX - (_d2 * i2))/scalethrw;

        oshiftz[i2*nz + i1] -= shiftz[i2*nz + i1]/_d1;
        oshiftx[i2*nz + i1] -= shiftx[i2*nz + i1]/_d2;

        // Save new label out
        nlblot[i2*nz + i1] *= fabsf(shiftz[i2*nz + i1]);
      }
    }
    bool found = false;
    int i1 = 0;
    while (i1 < nz - 1 && !found) {
      if (fabs(shiftz[i2*nz + i1]) > 0.)
        found = true;
      else
        i1++;
    }
    if (found) {
      for (int i = 0; i <= i1; i++) {
        shiftz[i2*nz + i] = shiftz[i2*nz + i1];
      }
    }
  }

  /* Free memory */
  delete[] shiftz; delete[] shiftx;
}

void evntcre8::squish(int nz, int *lyrin, float *velin, float *shftin, int mode,
    float azim, float maxshift, float lambda, float rinline, float rxline,
    int nzot, int *lyrot, float *velot) {

  /* Compute model lengths */
  float dst2 = _d2 * _n2;
  float dst3 = _d3 * _n3;

  /* Allocate shift and dist arrays */
  int nn = std::max(_n2, _n3) * 3;
  float *shift = new float[nn*nn]();
  float *dist  = new float[nn*nn]();

  // Maximum shift in samples (nzot should be nzin + 2*iMaxShift)
  int iMaxShift = maxshift / _d1;
  if(nzot != nz + 2*iMaxShift) {
    fprintf(stderr,"squish: Ouptut nz must be nzin + 2*maxshift/dz\n");
    exit(EXIT_FAILURE);
  }

  float wavelength = std::max(dst2, dst3)*lambda;

  /* Use cosine to compute the shift action */
  if(mode == 0) {
    /* Compute shift and dist arrays */
    float w[] = {1.2, 5., 1.9, 2.4, 3.8, 1.4, 4.2, 3.1, 3.4, 2.0};
    std::vector<float> wv(w, w + sizeof(w) / sizeof(float));
    for (auto a = wv.begin(); a != wv.end(); ++a) *a *= wavelength;
    float pi = atan(1.) * 4;
    for (int i2 = 0; i2 < nn; i2++) {
      for (int i1 = 0; i1 < nn; i1++) {
        shift[i2*nn + i1] = cosf(_d2 / wavelength * pi * i2);
      }
    }

    /* Add some randomness to shift */
    std::vector<float> rands(20);
    for (int i = 0; i < 20; i++) {
      rands[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - .5) * 2.;
    }

    float mxv = 0;
    for (int iw = 0; iw < 10; iw++) {
      float d2 = 0;
      for (int i3 = 0; i3 < nn; i3++) {
        float dd = cosf(d2 / wv[iw] * pi) * rands[10 + iw] * rxline;
        float d = 0;
        for (int i2 = 0; i2 < nn; i2++) {
          shift[i3*nn + i2] += dd * cosf(d / wv[iw] * pi) * rands[iw] * rinline;

          if (iw == 9) mxv = std::max(mxv, fabsf(shift[i3*nn + i2]));
          d += _d2;
        }
      }
    }

    /* Scale by max shift */
    for (int i3 = 0; i3 < nn; i3++) {
      for (int i2 = 0; i2 < nn; i2++) {
        shift[i3*nn + i2] = maxshift * shift[i3*nn + i2] / mxv;
      }
    }

  } else{
    /* Copy input shift function */
    memcpy(shift,shftin,sizeof(float)*nn*nn);
  }

  /* Rotate shift into azimuth */
  float *shiftrot = new float[_n2*_n3]();
  float *distrot  = new float[_n2*_n3]();

  float cs = cosf(azim * atan(1.) / 45.);
  float sn = sinf(azim * atan(1.) / 45.);

  rotate_array(cs, sn, nn, shift, _n2, _d2, _n3, _d3, shiftrot);
  if(mode == 0) {
    calcshift_fastaxis(nn, _d2, shift, dist);
    rotate_array(cs, sn, nn, dist , _n2, _d2, _n3, _d3, distrot );
  }

  /* Create top and bottom arrays */
  float *top = new float[_n2*_n3]();
  float *bot = new float[_n2*_n3]();

  /* Apply random shifts for cosine mode */
  if(mode == 0) {
    fill_random(_n2, _n3, top);
    fill_random(_n2, _n3, bot);
    for(int k = 0; k < 3; ++k) {
      smooth(_n2, _n3, top, 25, 25);
      smooth(_n2, _n3, bot, 25, 25);
    }
    scale(_n2, _n3, top, 0.45); scale(_n2, _n3, bot, 0.45);
    add  (_n2, _n3, top, 0.55); add  (_n2, _n3, bot, 0.55);
  }

  /* Apply shifts */
  for (int i3 = 0; i3 < _n3; i3++) {
    for (int i2 = 0; i2 < _n2; i2++) {
      int ib = std::min(std::max(i3 + (int)(sn * shiftrot[i3*_n2 + i2] / _d3), 0), _n3 - 1);
      int ia = std::min(std::max(i2 + (int)(cs * shiftrot[i3*_n2 + i2] / _d3), 0), _n2 - 1);

      int i1 = shiftrot[i3*_n2 + i2] / _d1; // Rotated shift to be applied
      int ibeg = iMaxShift + i1;            // Loop up to maximum shift
      for (int i = 0; i < ibeg; i++) {
        lyrot[i3*nzot*_n2 + i2*nzot + i] = -1;
        velot[i3*nzot*_n2 + i2*nzot + i] = -1;
      }

      for (i1 = 0; i1 < nz; i1++) {
        float f2 = (float)(i1 - 1) / (float)nz;
        float f3 = top[i3*_n2 + i2]*(1.-f2) + f2 * bot[i3*_n2 + i2];

        f3 *= distrot[i3*_n2 + i2];

        int ix = i1 - (int)(f3 / _d1);

        if (ix < 1) {
          lyrot[i3*nzot*_n2 + i2*nzot + ibeg+i1] = -1;
          velot[i3*nzot*_n2 + i2*nzot + ibeg+i1] = -1;
        } else {
          ix = std::min(ix, nz - 1);
          lyrot[i3*nzot*_n2 + i2*nzot + ibeg+i1] = lyrin[i3*nz*_n2 + i2*nz + ix];
          velot[i3*nzot*_n2 + i2*nzot + ibeg+i1] = velin[i3*nz*_n2 + i2*nz + ix];
        }
      }
      for (int i1 = nz + ibeg; i1 < nzot; i1++) {
        lyrot[i3*nzot*_n2 + i2*nzot + i1] = lyrin[ib*nz*_n2 + ia*nz + nz-1];
        velot[i3*nzot*_n2 + i2*nzot + i1] = velin[ib*nz*_n2 + ia*nz + nz-1];
      }
    }
  }

  /* Free memory */
  delete[] shift;    delete[] dist;
  delete[] shiftrot; delete[] distrot;
  delete[] top;      delete[] bot;

}

void evntcre8::squish_shifts(int nz, float *shftin, int mode,
    float azim, float lambda, float rinline, float rxline,
    float *shfty, float *shftx, float *shftz) {
  /* Compute model lengths */
  float dst2 = _d2 * _n2;
  float dst3 = _d3 * _n3;

  /* Allocate shift and dist arrays */
  int nn = std::max(_n2, _n3) * 3;
  float *shift = new float[nn*nn]();
  float *dist  = new float[nn*nn]();

  float wavelength = std::max(dst2, dst3)*lambda;

  /* Use cosine to compute the shift action */
  if(mode == 0) {
    /* Compute shift and dist arrays */
    float w[] = {1.2, 5., 1.9, 2.4, 3.8, 1.4, 4.2, 3.1, 3.4, 2.0};
    std::vector<float> wv(w, w + sizeof(w) / sizeof(float));
    for (auto a = wv.begin(); a != wv.end(); ++a) *a *= wavelength;
    float pi = atan(1.) * 4;
    for (int i2 = 0; i2 < nn; i2++) {
      for (int i1 = 0; i1 < nn; i1++) {
        shift[i2*nn + i1] = cosf(_d2 / wavelength * pi * i2);
      }
    }

    /* Add some randomness to shift */
    std::vector<float> rands(20);
    for (int i = 0; i < 20; i++) {
      rands[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - .5) * 2.;
    }

    float mxv = 0;
    for (int iw = 0; iw < 10; iw++) {
      float d2 = 0;
      for (int i3 = 0; i3 < nn; i3++) {
        float dd = cosf(d2 / wv[iw] * pi) * rands[10 + iw] * rxline;
        float d = 0;
        for (int i2 = 0; i2 < nn; i2++) {
          shift[i3*nn + i2] += dd * cosf(d / wv[iw] * pi) * rands[iw] * rinline;

          if (iw == 9) mxv = std::max(mxv, fabsf(shift[i3*nn + i2]));
          d += _d2;
        }
      }
    }

//    /* Scale by max shift */
//    for (int i3 = 0; i3 < nn; i3++) {
//      for (int i2 = 0; i2 < nn; i2++) {
//        shift[i3*nn + i2] = maxshift * shift[i3*nn + i2] / mxv;
//      }
//    }

  } else{
    /* Copy input shift function */
    memcpy(shift,shftin,sizeof(float)*nn*nn);
  }

  /* Rotate shift into azimuth */
  float *shiftrot = new float[_n2*_n3]();
  float *distrot  = new float[_n2*_n3]();

  float cs = cosf(azim * atan(1.) / 45.);
  float sn = sinf(azim * atan(1.) / 45.);

  rotate_array(cs, sn, nn, shift, _n2, _d2, _n3, _d3, shiftrot);
  if(mode == 0) {
    calcshift_fastaxis(nn, _d2, shift, dist);
    rotate_array(cs, sn, nn, dist , _n2, _d2, _n3, _d3, distrot );
  }

  /* Create top and bottom arrays */
  float *top = new float[_n2*_n3]();
  float *bot = new float[_n2*_n3]();

  /* Apply random shifts for cosine mode */
  if(mode == 0) {
    fill_random(_n2, _n3, top);
    fill_random(_n2, _n3, bot);
    for(int k = 0; k < 3; ++k) {
      smooth(_n2, _n3, top, 25, 25);
      smooth(_n2, _n3, bot, 25, 25);
    }
    scale(_n2, _n3, top, 0.45); scale(_n2, _n3, bot, 0.45);
    add  (_n2, _n3, top, 0.55); add  (_n2, _n3, bot, 0.55);
  }

  /* Apply shifts */
  for (int i3 = 0; i3 < _n3; i3++) {
    for (int i2 = 0; i2 < _n2; i2++) {

      float beg = shiftrot[i3*_n2 + i2]; // Rotated shift to be applied

      //TODO: will need to bring back distrot for cosine shifts

      for (int i1 = 0; i1 < nz; i1++) {
        shftz[i3*nz*_n2 + i2*nz + i1] = (i1*_d1 - beg)/_d1;
        shftx[i3*nz*_n2 + i2*nz + i1] = i2;
        shfty[i3*nz*_n2 + i2*nz + i1] = i3;
      }
    }
  }

  /* Free memory */
  delete[] shift;    delete[] dist;
  delete[] shiftrot; delete[] distrot;
  delete[] top;      delete[] bot;
}

void evntcre8::fill_top_bottom(int nz, float maxshift,
    float basevel, int *lyrot, float *velot) {

  float velup = -1.0;

  for(int i3 = 0; i3 < _n3; ++i3) {
    for(int i2 = 0; i2 < _n2; ++i2) {
      for(int i1 = 0; i1 < nz-1; ++i1) {
        /* First find the top interface */
        int diff1 = lyrot[i3*nz*_n2 + i2*nz + i1+1] - lyrot[i3*nz*_n2 + i2*nz + i1];
        if(diff1 == 2) {
          /* Grab a sample 10 samples down and continue that up */
          if(velup == -1.0) {
            velup = velot[i3*nz*_n2 + i2*nz + i1+10]; //XXX: Assumes that the layer is > 10 samples thick
          }
          for(int it = 0; it < 8; ++it) {
            velot[i3*nz*_n2 + i2*nz + i1+5-it] = velup;
            lyrot[i3*nz*_n2 + i2*nz + i1+5-it] = 1;
          }
        }

        /* Handle the bottom-most interface */
        int diff2 = lyrot[i3*nz*_n2 + i2*nz + i1+1] - lyrot[i3*nz*_n2 + i2*nz + i1];
        if(diff2 == -1) {
          for(int ib = i1; ib < nz; ++ib) {
            velot[i3*nz*_n2 + i2*nz + ib] = basevel;
            lyrot[i3*nz*_n2 + i2*nz + ib] = 0;
          }
        }
      }
    }
  }
}

int evntcre8::find_max_deposit(int n1, int n2, int n3, int *lyrin) {
  int mx = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, n3), int(0),
      [&](tbb::blocked_range<size_t> r, float value) -> float {
    for (size_t i3 = r.begin(); i3 != r.end(); i3++) {
      for (int i2 = 0; i2 < n2; i2++) {
        int i1 = 0;
        while (i1 < n1 - 1 && lyrin[i3*n1*n2 + i2*n1 + i1] == -1) i1++;
        if (value < i1) value = i1;
      }
    }
    return value;
  },
  [](int a, int b) {
    if (a > b) return a;
    return b;
  });
  return mx + 1;
}

void evntcre8::fill_random(int n1, int n2, float *velin) {
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, n2),
      [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i2 = r.begin(); i2 != r.end(); ++i2) {
      for (int i1 = 0; i1 < n1; ++i1) {
        velin[i2*n1 + i1] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) - 0.5;
      }
    }
  });

}

void evntcre8::fill_random(int n1, int n2, int n3, float *velin) {
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, n3),
      [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i3 = r.begin(); i3 != r.end(); ++i3) {
      for (int i2 = 0; i2 < n2; ++i2) {
        for (int i1 = 0; i1 < n1; ++i1) {
          velin[i3*n1*n2 + i2*n1 + i1] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) - 0.5;
        }
      }
    }
  });
}

void evntcre8::bandpass(float f1, float f2, float f3, int n1, int n2, int n3, float *velin) {

  std::vector<int> n(3, n1), nc(3, 1);

  n[1] = n2;
  n[2] = n3;
  nc[0] = n[0] / 2 + 1;
  nc[1] = n[1];
  nc[2] = n[2];
  size_t n123 = nc[0] * nc[1] * nc[2];

  fftwf_complex* tmp = new fftwf_complex[n123];
  fftwf_init_threads();
  std::vector<float> sc1 = find_lim1(nc[0], f1);
  std::vector<float> sc2 = find_lim2(nc[1], f2);
  std::vector<float> sc3 = find_lim2(nc[2], f3);

  fftwf_plan_with_nthreads(16);
  fftwf_plan forP = fftwf_plan_dft_r2c_3d(n[2], n[1], n[0], velin, tmp, FFTW_ESTIMATE);
  fftwf_plan bckP = fftwf_plan_dft_c2r_3d(n[2], n[1], n[0], tmp, velin, FFTW_ESTIMATE);

  size_t n12 = nc[0] * nc[1];
  float iN123 = 1. / (float)(n[0] * n[1] * n[2]);
  fftwf_execute(forP);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, nc[2]),
      [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i3 = r.begin(); i3 < r.end(); i3++) {
      for (int i2 = 0; i2 < nc[1]; i2++) {
        for (int i1 = 0; i1 < nc[0]; i1++) {
          float sc = sc1[i1] * sc2[i2] * sc3[i3];
          tmp[i1 + i2 * nc[0] + i3 * n12][1] *= sc * iN123;
          tmp[i1 + i2 * nc[0] + i3 * n12][0] *= sc * iN123;
          ;
        }
      }
    }
  });

  fftwf_execute(bckP);

  delete[] tmp;
  fftwf_destroy_plan(forP);
  fftwf_destroy_plan(bckP);
  fftwf_cleanup_threads();
}

std::vector<float> evntcre8::find_lim2(int n, float f) {
  int b = std::min(n / 2 - 4, (int)(f * n / 2));
  int e = n - b;
  float cs[4] = {.9239, .7071, .3827, .0};
  std::vector<float> sc(n, 0.);
  for (int i = 0; i < b; i++) sc[i] = 1;
  for (int i = 0; i < 4; i++) sc[i + b] = cs[i];
  for (int i = 0; i < 4; i++) sc[e - 3 + i] = cs[3 - i];
  for (int i = e; i < n; i++) sc[i] = 1.;
  return sc;
}

std::vector<float> evntcre8::find_lim1(int n, float f) {
  int b = std::min(n - 4, (int)(f * n));
  float cs[4] = {.9239, .7071, .3827, .0};
  std::vector<float> sc(n, 0.);
  for (int i = 0; i < b; i++) sc[i] = 1;
  for (int i = 0; i < 4; i++) sc[i + b] = cs[i];
  return sc;
}

float evntcre8::find_max(int n1, int n2, int n3, float *arr) {

  float mx = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, n3), float(0.0),
      [&](tbb::blocked_range<size_t> r, float value) -> float {
    for (size_t i3 = r.begin(); i3 != r.end(); i3++) {
      for (int i2 = 0; i2 < n2; i2++) {
        for (int i1 = 0; i1 < n1; i1++) {
          if (arr[i3*n1*n2 + i2*n1 + i1] > value) value = arr[i3*n1*n2 + i2*n1 + i1];
        }
      }
    }
    return value;
  },
  [](float a, float b) {
    if (a > b) return a;
    return b;
  });
  return mx;

}

float evntcre8::find_min(int n1, int n2, int n3, float *arr) {

  float mn = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, n3), float(0.0),
      [&](tbb::blocked_range<size_t> r, float value) -> float {
    for (size_t i3 = r.begin(); i3 != r.end(); i3++) {
      for (int i2 = 0; i2 < n2; i2++) {
        for (int i1 = 0; i1 < n1; i1++) {
          if (arr[i3*n1*n2 + i2*n1 + i1] < value) value = arr[i3*n1*n2 + i2*n1 + i1];
        }
      }
    }
    return value;
  },
  [](float a, float b) {
    if (a < b) return a;
    return b;
  });
  return mn;
}

float evntcre8::find_absmax(int n1, int n2, int n3, float *arr) {
  return std::max(fabsf(find_max(n1,n2,n3,arr)), fabsf(find_min(n1,n2,n3,arr)));
}

void evntcre8::scale(int n1, int n2, int n3, float *arr, float sc) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n3),
      [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i3 = r.begin(); i3 != r.end(); ++i3) {
      for (int i2 = 0; i2 < n2; i2++) {
        for (int i1 = 0; i1 < n1; i1++) {
          arr[i3*n1*n2 + i2*n1 + i1] *= sc;
        }
      }
    }

  });
}

void evntcre8::scale(int n1, int n2, float *arr, float sc) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n2),
      [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i2 = r.begin(); i2 != r.end(); ++i2) {
      for (int i1 = 0; i1 < n1; i1++) {
        arr[i2*n1 + i1] *= sc;
      }
    }
  });
}

void evntcre8::add(int n1, int n2, float *arr, float ad) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n2),
      [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i2 = r.begin(); i2 != r.end(); ++i2) {
      for (int i1 = 0; i1 < n1; i1++) {
        arr[i2*n1 + i1] += ad;
      }
    }
  });
}

void evntcre8::norm(int n1, int n2, int n3, float *arr, float sc) {
  float mx = find_absmax(n1,n2,n3,arr);
  scale(n1,n2,n3,arr,sc/mx);
}

void evntcre8::zder(int nz, float *lblin, float *lblot) {

  for(int i3 = 0; i3 < _n3; ++i3) {
    for(int i2 = 0; i2 < _n2; ++i2) {
      for(int i1 = 0; i1 < nz; ++i1) {
        /* One-sided derivatives at the ends */
        if(i1 == 0) {
          lblot[i3*nz*_n2 + i2*nz + i1] = lblin[i3*nz*_n2 + i2*nz + i1+1] - lblin[i3*nz*_n2 + i2*nz + i1-0];
        } else if(i1 == nz-1) {
          lblot[i3*nz*_n2 + i2*nz + i1] = lblin[i3*nz*_n2 + i2*nz + i1+0] - lblin[i3*nz*_n2 + i2*nz + i1-1];
        } else {
          lblot[i3*nz*_n2 + i2*nz + i1] = 0.5*lblin[i3*nz*_n2 + i2*nz + i1+1] - 0.5*lblin[i3*nz*_n2 + i2*nz + i1-1];
        }
      }
    }
  }

}

void evntcre8::calcref(int nz, float *vel, float *ref) {

  for(int i3 = 0; i3 < _n3; ++i3) {
    for(int i2 = 0; i2 < _n2; ++i2) {
      for(int i1 = 0; i1 < nz; ++i1) {
        /* Backwards derivatives at the end */
        if(i1 == nz-1) {
          ref[i3*nz*_n2 + i2*nz + i1] = vel[i3*nz*_n2 + i2*nz + i1  ] - vel[i3*nz*_n2 + i2*nz + i1-1];
        } else {
          ref[i3*nz*_n2 + i2*nz + i1] = vel[i3*nz*_n2 + i2*nz + i1+1] - vel[i3*nz*_n2 + i2*nz + i1  ];
        }
      }
    }
  }
}

void evntcre8::calcref2d(int nx, int nz, float *vel, float *ref) {
  for(int i2 = 0; i2 < nx; ++i2) {
    for(int i1 = 0; i1 < nz; ++i1) {
      /* Backwards derivatives at the end */
      if(i1 == nz-1) {
        ref[i2*nz + i1] = vel[i2*nz + i1  ] - vel[i2*nz + i1-1];
      } else {
        ref[i2*nz + i1] = vel[i2*nz + i1+1] - vel[i2*nz + i1  ];
      }
    }
  }
}

void evntcre8::laplacian(int nz, float *lblin, float *lblot) {

  for(int i3 = 1; i3 < _n3-1; ++i3) {
    for(int i2 = 1; i2 < _n2-1; ++i2) {
      for(int i1 = 1; i1 < nz-1; ++i1) {
        lblot[i3*nz*_n2 + i2*nz + i1] = -6*lblin[i3*nz*_n2 + i2*nz + i1] +
            lblin[(i3-1)*nz*_n2 + (i2  )*nz + i1  ] +  lblin[(i3+1)*nz*_n2 + (i2  )*nz + i1  ] +
            lblin[(i3  )*nz*_n2 + (i2-1)*nz + i1  ] +  lblin[(i3  )*nz*_n2 + (i2+1)*nz + i1  ] +
            lblin[(i3  )*nz*_n2 + (i2  )*nz + i1-1] +  lblin[(i3  )*nz*_n2 + (i2  )*nz + i1+1];
      }
    }
  }

}

void evntcre8::calcshift_fastaxis(int nn, float d2, float *shift, float *dist) {

  int center2 = nn / 2 * d2;
  float *bb = new float[nn*nn]();
  for (int i3 = 0; i3 < nn; i3++) {
    float d = 0;
    for (int i2 = 0; i2 < nn - 1; i2++) {
      d += d2 / cosf(atanf( (shift[i3*nn + i2+1] - shift[i3*nn + i2]) / d2 ));
      bb[i3*nn + i2] = d;
    }
    d += d2 / cosf(atanf( (shift[i3*nn + nn-1] - shift[i3*nn + nn-2]) / d2 ));
    bb[i3*nn + nn-1] = d;
  }

  for (int i3 = 0; i3 < nn; i3++) {
    float dd = bb[i3*nn + nn/2];
    float d  = 0;
    for (int i2 = 0; i2 < nn; i2++) {
      d += d2;
      dist[i3*nn + i2] = bb[i3*nn + i2] - dd + center2 - d;
    }
  }

  delete[] bb;

}

void evntcre8::rotate_array(float cs, float sn, int nn, float *shift, int n2, float d2, int n3, float d3, float *shiftrot) {

  tbb::parallel_for(
      tbb::blocked_range<int>(0, _n3), [&](const tbb::blocked_range<int>& r) {
    for (int i3 = r.begin(); i3 != r.end(); ++i3) {
      float p3 = ((float)(i3 - _n3) / 2.) * _d3;
      for (int i2 = 0; i2 < _n2; i2++) {
        float p2 = (i2 - _n2 / 2) * _d2;
        /* Multiply with rotation matrix */
        float d  = cs * p2 - sn * p3;
        float d2 = sn * p2 + cs * p3;
        /* Linear interpolation */
        float f2 = d  / _d2 + nn / 2;
        float f3 = d2 / _d2 + nn / 2;
        int ia = f2;
        int ib = f3;
        f2 -= ia;
        f3 -= ib;

        if (ib >= 0 && ib < nn - 1 && ia >= 0 && ia < nn - 1) {
          shiftrot[i3*n2 + i2] = (1.-f2)*(1.-f3)*shift[ib*nn + ia] + (1.-f2)*f3*shift[ib*nn + ia+1] +
              f2*(1.-f3)*shift[ib*nn + ia+1] + f2*f3*shift[(ib+1)*nn + ia+1];
        } else
          shiftrot[i3*n2 + i2] = 0;
      }
    }
  });
}

void evntcre8::smooth(int n1, int n2, float *arr, int len1, int len2) {

  tbb::parallel_for(tbb::blocked_range<size_t>(0, n2),
      [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i2 = r.begin(); i2 != r.end(); ++i2) {
      std::vector<float> vec1(2 * len1 + n1, 0.), vec2(2 * len1 + n1, 0.);

      for (int i1 = 0; i1 < n1; i1++) {
        vec1[i1 + len1] = arr[i2*n1 + i1];
      }

      recForward(vec1, vec2, len1);
      recBackward(vec2, vec1, len1);

      for (int i1 = 0; i1 < n1; i1++) {
        arr[i2*n1 + i1] = vec1[i1 + len1];
      }
    }

  });

  tbb::parallel_for(tbb::blocked_range<size_t>(0, n1),
      [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i1 = r.begin(); i1 != r.end(); ++i1) {
      std::vector<float> vec1(2 * len2 + n2, 0.), vec2(2 * len2 + n2, 0.);

      for (int i2 = 0; i2 < n2; i2++) {
        vec1[i2 + len2] = arr[i2*n1 + i1];
      }

      recForward(vec1, vec2, len2);
      recBackward(vec2, vec1, len2);

      for (int i2 = 0; i2 < n2; i2++) {
        arr[i2*n1 + i1] = vec1[i2 + len2];
      }
    }
  });
}

void evntcre8::recForward(std::vector<float>& vecIn, std::vector<float>& vecOut, const int len) {
  float t = 0;
  float mul = 1. / (float)(2 * len + 1.);

  for (int i = 0; i < len; i++) {
    vecIn[i] = vecIn[len];
    vecIn[vecIn.size() - i - 1] = vecIn[vecIn.size() - 1 - len];
  }

  for (int i = 0; i < 2 * len; i++) t += vecIn[i];
  size_t n = vecIn.size() - 2 * len;

  for (size_t i = 0; i < n; i++) {
    t += vecIn[i + 2 * len];
    vecOut[i + len] = t * mul;
    t -= vecIn[i];
  }

}

void evntcre8::recBackward(std::vector<float>& vecIn, std::vector<float>& vecOut, const int len) {
  float t = 0;
  float mul = 1. / (float)(2 * len + 1.);

  for (int i = 0; i < len; i++) {
    vecIn[i] = vecIn[len];
    vecIn[vecIn.size() - i - 1] = vecIn[vecIn.size() - 1 - len];
  }

  for (int i = 0; i < 2 * len; i++) t += vecIn[vecIn.size() - 2 * len + i];
  size_t n = vecIn.size() - 2 * len;

  for (int i = n - 1; i >= 0; i--) {
    t += vecIn[i];
    vecOut[i + len] = t * mul;
    t -= vecIn[i + 2 * len];
  }

}
