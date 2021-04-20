"""
Functions for computing structural similarity index metrix (SSIM)
@author: Joseph Jennings
@version: 2020.04.12
"""
import numpy as np
from numpy.ma.core import exp
from skimage.measure import compare_ssim
import scipy.ndimage
from scipy import signal

def ssim(img,tgt,dr=None):
  """
  Computes the structural similarity index between two images

  Parameters
    img - the input image
    tgt - the target image (reference)
    dr  - the dynamic range [img.max() - img.min()]

  Returns the SSIM
  """
  return compare_ssim(img,tgt,dynamic_range=img.max() - img.min())

def cwssim(image,target,width=30,k=0.01):
  """
  Computes the complex wavelet SSIM

  Parameters
    image  - the input image to be evaluated
    target - the target image for comparison
    width  - width of ricker used for wavelet decomposition

  Returns the structural similarity index metric
  """
  # Flatten to perform a 1D CWT
  img = image.flatten(); tgt = target.flatten()
  # Compute wavelet widths
  widths = np.arange(1, width+1)
  
  # Convolution
  cwtmatr1 = signal.cwt(img, signal.ricker, widths)
  cwtmatr2 = signal.cwt(tgt, signal.ricker, widths)

  # Compute the first term
  c1c2 = np.multiply(abs(cwtmatr1), abs(cwtmatr2))
  c1_2 = np.square(abs(cwtmatr1))
  c2_2 = np.square(abs(cwtmatr2))
  num_ssim_1 = 2 * np.sum(c1c2, axis=0) + k
  den_ssim_1 = np.sum(c1_2, axis=0) + np.sum(c2_2, axis=0) + k

  # Compute the second term
  c1c2_conj = np.multiply(cwtmatr1, np.conjugate(cwtmatr2))
  num_ssim_2 = 2 * np.abs(np.sum(c1c2_conj, axis=0)) + k
  den_ssim_2 = 2 * np.sum(np.abs(c1c2_conj), axis=0) + k
  
  ssim_map = (num_ssim_1 / den_ssim_1) * (num_ssim_2 / den_ssim_2)

  # Average the per pixel results
  index = np.average(ssim_map)
  return index

def ssim2(img,tgt,width=11,sigma=1.5,k1=0.01,k2=0.03,l=255,k=0.01):

  # Compute constants
  c1 = (k1*l)**2
  c2 = (k2*l)**2

  # Generate gaussian kernel
  kern = gauss_kernel(width=width, sigma=sigma)
  
  # Estimate means of images
  mui  = conv_gauss2d(img, kern)
  mui2 = mui*mui
  mut  = conv_gauss2d(tgt, kern)
  mut2 = mut*mut

  # Estimate sigma squared of squared images
  sigi2  = conv_gauss2d(img*img, kern) - mui2
  sigt2  = conv_gauss2d(tgt*tgt, kern) - mut2

  # Compute the cross mean and variance
  cross  = img * tgt
  sigit  = conv_gauss2d(cross, kern)
  muit   = mui  * mut
  sigit -= muit 

  # Numerator of SSIM
  num = ((2 * muit + c1) * (2 * sigit + c2))

  # Denominator of SSIM
  den = ((mui2 + mut2 + c1) * (sigi2 + sigt2 + c2))
  
  ssim_map = num / den
  index = np.average(ssim_map)
  return index


def conv_gauss2d(image, kernel1d):
  """
  Convolve 2d gaussian
  """ 
  result = scipy.ndimage.filters.correlate1d(image, kernel1d, axis=0)
  result = scipy.ndimage.filters.correlate1d(result, kernel1d, axis=1)
  return result

def gauss_kernel(width=11, sigma=1.5):
  """ 
  Generate a gaussian kernel 
  """
  # 1D Gaussian kernel definition
  kernel = np.ndarray(width)
  norm_mu = int(width/ 2)
  
  # Fill Gaussian kernel
  for i in range(width):
    kernel [i] = (exp(-(((i - norm_mu) ** 2)) / (2 * (sigma ** 2))))

  return kernel / np.sum(kernel)

