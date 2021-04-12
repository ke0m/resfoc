"""
Metrics for quantitative evaluation of
deep learning predictions
@author: Joseph Jennings
@version: 2020.04.14
"""
import numpy as np
import matplotlib.pyplot as plt


def iou_score(target, prediction):
  intersection = np.logical_and(target, prediction)
  #plt.figure()
  #plt.imshow(intersection, cmap='jet')
  #plt.title('Intersection=%d' % (np.sum(intersection)))
  union = np.logical_or(target, prediction)
  #plt.figure()
  #plt.imshow(union, cmap='jet')
  #plt.title('Union=%d' % (np.sum(union)))
  #plt.show()
  return np.sum(intersection) / np.sum(union)


def dice_score(target, prediction):
  intersection = np.logical_and(target, prediction)
  return 2.0 * np.sum(intersection) / (np.sum(target) + np.sum(prediction))


def confusion_matrix(lbl, prd):
  # Label indices
  pos_lbl_idx = lbl == 1
  neg_lbl_idx = lbl == 0
  # Prediction indices
  pos_prd_idx = prd == 1
  neg_prd_idx = prd == 0
  # Number True positives
  tp = np.sum(np.logical_and(pos_lbl_idx, pos_prd_idx))
  tn = np.sum(np.logical_and(neg_lbl_idx, neg_prd_idx))


def intersect(a, b):
  """
  Computes the intersection between two thresholded numpy arrays

  Parameters
    a - input numpy array
    b - input numpy array

  Returns a numpy array containing the intersection between the two arrays
  """
  n = len(a)
  if (n != len(b)):
    raise Exception("Input arrays must be same length")
  aidx = a == 1.0
  bidx = b == 1.0
  return np.logical_and(aidx, bidx)


def union(a, b):
  pass
