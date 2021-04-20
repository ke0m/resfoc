"""
Utility functions for multi-dimensional
signal processing.
2D band and lowpass filtering functions
are from author Sam

@author: Joseph Jennings
@version: 2020.02.09
"""
import numpy as np
from skimage import exposure
from scipy import misc
from scipy.signal import butter, lfilter, filtfilt

def ampspec1d(sig,dt):
  """ 
  Returns the amplitude spectrum and the frequencies of the signal 

  Parameters:
    sig - the input signal (numpy array)
    dt  - the sampling rate of the signal
  """
  nt = sig.shape[0]
  spec = np.abs(np.fft.fftshift(np.fft.fft(np.pad(sig,(0,nt),mode='constant'))))[nt:]
  nf = nt; of = 0.0; df = 1/(2*dt*nf)
  fs = np.linspace(of,of+(nf-1)*df,nf)

  return spec,fs

def ampspec2d(img,d1,d2):
  """
  Returns the 2D amplitude spectrum of the image and the
  corresponding wavenumbers

  Parameters:
    img - the input image
    d1  - the sampling along the fast axis
    d2  - the sampling along the slow axis
  """
  n1 = img.shape[1]; dk1 = 1/(n1*d1); ok1 = -dk1*n1/2.0;
  n2 = img.shape[0]; dk2 = 1/(n2*d2); ok2 = -dk2*n2/2.0;
  k1 = np.linspace(ok1, ok1+(n1-1)*dk1, n1)
  k2 = np.linspace(ok2, ok2+(n2-1)*dk2, n2)
  imgfft = np.abs(np.fft.fftshift(np.fft.fft2(img)))

  return imgfft,k1,k2

def butter_bandpass(locut, hicut, fs, order=5):
  """
  Returns the numerator and demoninator of the transfer function
  of a Butterworth_bandpass filter.

  Parameters
    locut - the low frequency beyond which to not pass
    hicut - the hight frequency beyond which to not pass
    fs    - the sampling frequency
    order - the order (number of terms in the LCCDE) to use for the filter

  @source: SO: how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
  @author: Warren Weckesser
  """
  nyq = 0.5 * fs
  lo = locut / nyq
  hi = hicut / nyq
  b, a = butter(order, [lo, hi], btype='band')
  return b, a

def butter_bandpass_filter(data, locut, hicut, fs, order=5):
  """
  Convolves the input signal with a butterworth bandpass filter

  Parameters
    locut - the low frequency beyond which to not pass
    hicut - the hight frequency beyond which to not pass
    fs    - the sampling frequency
    order - the order (number of terms in the LCCDE) to use for the filter

  @source: SO: how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
  @author: Warren Weckesser
  """
  b, a = butter_bandpass(locut, hicut, fs, order=order)
  y = filtfilt(b, a, data)
  return y

def butter2d_lp(shape, f, n, pxd=1):
  """Designs an n-th order lowpass 2D Butterworth filter with cutoff
   frequency f. pxd defines the number of pixels per unit of frequency (e.g.,
   degrees of visual angle)."""
  pxd = float(pxd)
  rows, cols = shape
  x = np.linspace(-0.5, 0.5, cols)  * cols / pxd
  y = np.linspace(-0.5, 0.5, rows)  * rows / pxd
  radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
  filt = 1 / (1.0 + (radius / f)**(2*n))
  return filt

def butter2d_bp(shape, cutin, cutoff, n, pxd=1):
  """Designs an n-th order bandpass 2D Butterworth filter with cutin and
   cutoff frequencies. pxd defines the number of pixels per unit of frequency
   (e.g., degrees of visual angle)."""
  return butter2d_lp(shape,cutoff,n,pxd) - butter2d_lp(shape,cutin,n,pxd)

def butter2d_hp(shape, f, n, pxd=1):
  """Designs an n-th order highpass 2D Butterworth filter with cutin
   frequency f. pxd defines the number of pixels per unit of frequency (e.g.,
   degrees of visual angle)."""
  return 1. - butter2d_lp(shape, f, n, pxd)

def ideal2d_lp(shape, f, pxd=1):
  """Designs an ideal filter with cutoff frequency f. pxd defines the number
   of pixels per unit of frequency (e.g., degrees of visual angle)."""
  pxd = float(pxd)
  rows, cols = shape
  x = np.linspace(-0.5, 0.5, cols)  * cols / pxd 
  y = np.linspace(-0.5, 0.5, rows)  * rows / pxd 
  radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
  filt = np.ones(shape)
  filt[radius>f] = 0
  return filt

def ideal2d_bp(shape, cutin, cutoff, pxd=1):
  """Designs an ideal filter with cutin and cutoff frequencies. pxd defines
   the number of pixels per unit of frequency (e.g., degrees of visual
   angle)."""
  return ideal2d_lp(shape,cutoff,pxd) - ideal2d_lp(shape,cutin,pxd)

def ideal2d_hp(shape, f, n, pxd=1):
  """Designs an ideal filter with cutin frequency f. pxd defines the number
   of pixels per unit of frequency (e.g., degrees of visual angle)."""
  return 1. - ideal2d_lp(shape, f, n, pxd)

def bandpass(data, highpass, lowpass, n, pxd, eq=None):
  """Designs then applies a 2D bandpass filter to the data array. If n is
   None, and ideal filter (with perfectly sharp transitions) is used
   instead."""
  fft = np.fft.fftshift(np.fft.fft2(data))
  if n:
    H = butter2d_bp(data.shape, highpass, lowpass, n, pxd)
  else:
    H = ideal2d_bp(data.shape, highpass, lowpass, pxd)
  fft_new = fft * H
  new_image = np.real(np.fft.ifft2(np.fft.ifftshift(fft_new)))
  if eq == 'histogram':
    new_image = exposure.equalize_hist(new_image)
  return new_image

