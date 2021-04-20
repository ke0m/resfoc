"""
Deep learning utility functions.
Perform pre and post processing of training data
Also some plotting utlities

@author: Joseph Jennings
@version: 2020.10.07
"""
import sys
import numpy as np
from scipy import interpolate
from tqdm import tqdm
from .python_patch_extractor.PatchExtractor import PatchExtractor
from .ptyprint import create_inttag


def normalize(img, eps=sys.float_info.epsilon, mode='2d'):
  """
  Normalizes an image accross channels  by removing
  the mean and dividing by the standard deviation

  Parameters
    img - the input image. If the image has three dimensions,
          will normalize each image individually
    eps - parameters to avoid dividing a zero standard deviation
    mode - normalize in 2D or 3D (['2d'] or '3d')

  Returns normalized image(s)
  """
  if (mode == '2d'):
    if (len(img.shape) == 3):
      imgnrm = np.zeros(img.shape)
      nimg = img.shape[0]
      for k in range(nimg):
        imgnrm[k] = (img[k] - np.mean(img[k])) / (np.std(img[k]) + eps)
      return imgnrm
    else:
      return (img - np.mean(img)) / (np.std(img) + eps)
  elif (mode == '3d'):
    if (len(img.shape) == 4):
      imgnrm = np.zeros(img.shape)
      nimg = img.shape[0]
      for k in range(nimg):
        imgnrm[k] = (img[k] - np.mean(img[k])) / (np.std(img[k]) + eps)
      return imgnrm
    else:
      return (img - np.mean(img)) / (np.std(img) + eps)
  else:
    raise Exception("Mode not recognized")


def resizepow2(img, kind='linear'):
  """
  Resizes an image so that its dimensions are
  a power of 2
  """
  # Get input shape
  length = img.shape[-1]
  height = img.shape[-2]
  # Compute the new shape
  lengthnew = next_power_of_2(length)
  heightnew = next_power_of_2(height)
  new_shape = [heightnew, lengthnew]
  # Resample the image
  return resample(img, new_shape, kind)


def resample(img, new_shape, kind='linear', ds=[]):
  """
  Resamples an image. Can work up to 4D numpy arrays.
  assumes that the nz and nx axes are the last two (fastest)
  """
  if (len(img.shape) >= 2):
    # Original coordinates
    length = img.shape[-1]
    height = img.shape[-2]
    x = np.linspace(0, length, length)
    y = np.linspace(0, height, height)
    # New coordinates for interpolation
    xnew = np.linspace(0, length, new_shape[1])
    ynew = np.linspace(0, height, new_shape[0])
  elif (len(img.shape) == 1):
    length = img.shape[0]
    x = np.linspace(0, length, length)
    # New coordinates for interpolation
    xnew = np.linspace(0, length, new_shape)
  # Compute new samplings
  if (len(ds) != 0):
    dout = []
    lr = new_shape[1] / length
    hr = new_shape[0] / height
    dout.append(ds[1] / lr)
    dout.append(ds[0] / hr)
  # Perform the interpolation
  if len(img.shape) == 4:
    res = np.zeros([img.shape[0], img.shape[1], new_shape[0], new_shape[1]],
                   dtype='float32')
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        f = interpolate.interp2d(x, y, img[i, j, :, :], kind=kind)
        res[i, j, :, :] = f(xnew, ynew)
  elif len(img.shape) == 3:
    res = np.zeros([img.shape[0], new_shape[0], new_shape[1]], dtype='float32')
    for i in tqdm(range(img.shape[0]), ascii=True, desc="nimg:"):
      f = interpolate.interp2d(x, y, img[i, :, :], kind=kind)
      res[i, :, :] = f(xnew, ynew)
  elif len(img.shape) == 2:
    res = np.zeros([new_shape[0], new_shape[1]], dtype='float32')
    f = interpolate.interp2d(x, y, img, kind=kind)
    res[:] = f(xnew, ynew)
  elif len(img.shape) == 1:
    res = np.zeros(new_shape, dtype='float32')
    f = interpolate.interp1d(x, img, kind=kind)
    res[:] = f(xnew)

  if (len(ds) == 0):
    return res
  else:
    return res, dout


def next_power_of_2(x):
  """ Gets the nearest power of two to x """
  return 1 if x == 0 else 2**(x - 1).bit_length()


def thresh(arr, thresh, mode='gt', absval=True):
  """ Applies a threshold to an array """
  out = np.zeros(arr.shape, dtype='float32')
  if (mode == 'eq'):
    idx = arr == thresh
    out[idx] = 1
    out[~idx] = 0
  elif mode == 'gt':
    if absval:
      idx = np.abs(arr) > thresh
    else:
      idx = arr > thresh
    out[idx] = 1
    out[~idx] = 0
  elif mode == 'ge':
    if absval:
      idx = np.abs(arr) >= thresh
    else:
      idx = arr >= thresh
    out[idx] = 1
    out[~idx] = 0
  elif mode == 'lt':
    if absval:
      idx = np.abs(arr) < thresh
    else:
      idx = arr < thresh
    out[idx] = 1
    out[~idx] = 0
  elif mode == 'le':
    if absval:
      idx = np.abs(arr) <= thresh
    else:
      idx = arr <= thresh
    out[idx] = 1
    out[~idx] = 0

  return out


def normextract(
    img,
    nzp=64,
    nxp=64,
    strdz=None,
    strdx=None,
    norm=True,
    flat=True,
):
  """
  Extract patches from an image and normalize each patch. Works for 2D
  and for 3D when the third dimension stride is one.

  Parameters:
    img   - the input image [n3,nz,nx]
    nzp   - size of the patch in z dimension [64]
    nxp   - size of the patch in x dimension [64]
    strdz - size of patch stride in z dimension [32]
    strdx - size of patch stride in x dimension [32]
    norm  - normalize the patches [True]
    flat  - return the patches flattened [nptch,nzp,nxp] or in a
            grid [numpz,numpx,nzp,nxp]

    Returns normalized image patches
  """
  if (strdz is None):
    strdz = int(nzp / 2 + 0.5)
  if (strdx is None):
    strdx = int(nxp / 2 + 0.5)
  if (len(img.shape) == 2):
    # Extract patches
    pe = PatchExtractor((nzp, nxp), stride=(strdz, strdx))
    ptch = pe.extract(img)

    # Get patch dimensions
    numpz = ptch.shape[0]
    numpx = ptch.shape[1]

    # Flatten and normalize
    if (norm):
      ptchf = normalize(ptch.reshape([numpz * numpx, nzp, nxp]), mode='2d')
    else:
      ptchf = ptch.reshape([numpz * numpx, nzp, nxp])

  elif (len(img.shape) == 3):
    # Get size of third dimension
    n3 = img.shape[0]

    # Extract patches
    pea = PatchExtractor((n3, nzp, nxp), stride=(1, strdz, strdx))
    ptch = np.squeeze(pea.extract(img))

    # Get patch dimensions
    numpz = ptch.shape[0]
    numpx = ptch.shape[1]

    # Flatten and normalize
    if (norm):
      ptchf = normalize(ptch.reshape([numpz * numpx, n3, nzp, nxp]), mode='3d')
    else:
      ptchf = ptch.reshape([numpz * numpx, n3, nzp, nxp])
  else:
    raise Exception("function supported only up to 3D")

  return ptchf


def torchprogress(
    cur,
    bsz,
    tot,
    loss,
    acc,
    size=40,
    file=sys.stdout,
    save=True,
):
  """
  Prints a progress bar during training of a torch
  neural network

  Parameters:
    cur  - index of the current batch
    bsz  - size of a batch
    tot  - the total number batches
    loss - the current running loss value
    acc  - the current accuracy
    save - flag indicating whether to return the computed values

  Prints a progressbar to the screen
  """
  x = int(size * cur / tot)
  if (cur == 0):
    div = 1
  else:
    div = cur
  curform = create_inttag(cur, tot)
  loss /= div
  acc /= ((cur + 1) * bsz)
  fmt = "%s/%d [%s%s] loss=%.4g acc=%.4f\r"
  file.write(fmt % (curform, tot, "#" * x, "." * (size - x), loss, acc))
  if (cur == tot):
    file.write("\n")
  file.flush()
  if (save):
    return loss, acc


def plot_patchgrid2d(
    img,
    nzp,
    nxp,
    strdz=None,
    strdx=None,
    dz=None,
    dx=None,
    oz=None,
    ox=None,
    transp=False,
    pltcoords=True,
    **kwargs,
) -> None:
  """
  Plots the patch grid on the input image

  Parameters:
    img       - the input image [nz,nx]
    nzp       - the size of the patch in z (samples)
    nxp       - the size of the patch in x (samples)
    strdz     - patch stride in z (samples) [nzp//2]
    strdx     - patch stride in x (samples) [nxp//2]
    dz        - depth sampling of image [1.0]
    dx        - lateral sampling of image [1.0]
    oz        - depth origin of image [0.0]
    ox        - lateral origin of image [0.0]
    transp    - flag indicating to transpose the input image [False]
    pltcoords - flag indicating to plot the patch
                coordinates on the image [True]
  """
  # Get the image axes
  if (transp):
    nx, nz = img.shape
  else:
    nz, nx = img.shape
  if (dz is None):
    dz = 1.0
  if (oz is None):
    oz = 0.0
  if (dx is None):
    dx = 1.0
  if (ox is None):
    ox = 0.0

  if (strdz is None):
    strdz = nzp // 2
  if (strdx is None):
    strdx = nxp // 2

  # Do the patch extraction
  totptchs = normextract(img, nzp, nxp, strdz, strdx).shape[0]

  # Make the patch grids
  bgz = 0
  egz = nz * dz + 1
  dgz = nzp * dz
  bgx = 0
  egx = nx * dx + 1
  dgx = nxp * dx

  # Get number of patches in each dimensions
  nptchz, remz = divmod(nz, nzp)
  nptchx, remx = divmod(nx, nxp)

  # Plotting parameters
  cmap = kwargs.get('cmap', 'gray')
  pclip = kwargs.get('pclip', 1.0)
  vmin = kwargs.get('vmin', pclip * np.min(img))
  vmax = kwargs.get('vmax', pclip * np.max(img))
  xlabel = kwargs.get('xlabel', 'X (km)')
  zlabel = kwargs.get('zlabel', 'Z (km)')
  fsize = kwargs.get('fsize', 15)
  interp = kwargs.get('interp', 'bilinear')
  aspect = kwargs.get('aspect', 'auto')
  xmin = ox
  xmax = ox + nx * dx
  zmin = oz
  zmax = oz + nz * dz
  textsize = kwargs.get('textsize', 12)
  if (strdx != 0):
    xshft = 2
  if (strdz != 0):
    zshft = 2

  fig = plt.figure(figsize=(kwargs.get('wbox', 10), kwargs.get('hbox', 6)))
  ax = fig.gca()
  zticks = np.arange(bgz, egz, dgz)
  xticks = np.arange(bgx, egx, dgx)
  ax.set_xticks(xticks)
  ax.set_yticks(zticks)
  ax.imshow(img,
            cmap=cmap,
            interpolation=interp,
            vmin=vmin,
            vmax=vmax,
            extent=[xmin, xmax, zmax, zmin],
            aspect=aspect)
  ax.grid(linestyle='-', color='k', linewidth=2)
  ax.set_ylabel(zlabel, fontsize=fsize)
  ax.set_xlabel(xlabel, fontsize=fsize)
  ax.tick_params(labelsize=fsize)
  tot1 = nptchz * nptchx
  ax.set_title(
      r'Grid 1: X-stride=0, Z-stride=0 $\rightarrow \,\, %d\times%d = %d$ patches'
      % (nptchx, nptchz, tot1),
      fontsize=fsize)
  # Plot the coordinates
  if (pltcoords):
    plot_patchcoords(nptchz, nptchx, zticks, xticks, zmax, xmax, 0, 0, zshft,
                     xshft, textsize, 'k')

  if (strdx != 0):
    fig = plt.figure(figsize=(kwargs.get('wbox', 10), kwargs.get('hbox', 6)))
    ax = fig.gca()
    zticks = np.arange(bgz, egz, dgz)
    xticks = np.arange(bgx + strdx * dx, egx - strdx * dx, dgx)
    ax.set_xticks(xticks)
    ax.set_yticks(zticks)
    ax.imshow(img,
              cmap=cmap,
              interpolation=interp,
              vmin=vmin,
              vmax=vmax,
              extent=[xmin, xmax, zmax, zmin],
              aspect=aspect)
    ax.grid(which='major', linestyle='-', color='r', linewidth=2)
    ax.set_ylabel(zlabel, fontsize=fsize)
    ax.set_xlabel(xlabel, fontsize=fsize)
    ax.tick_params(labelsize=fsize)
    tot2 = (nptchx - 1) * nptchz
    ax.set_title(
        r'Grid 2: X-stride=%d, Z-stride=0 $\rightarrow \,\, %d\times%d = %d$ patches'
        % (strdx, nptchx - 1, nptchz, tot2),
        fontsize=fsize)
    if (pltcoords):
      plot_patchcoords(nptchz, nptchx - 1, zticks, xticks, zmax, xmax, 0, 1,
                       zshft, xshft, textsize, 'r')

  if (strdz != 0):
    fig = plt.figure(figsize=(kwargs.get('wbox', 10), kwargs.get('hbox', 6)))
    ax = fig.gca()
    zticks = np.arange(bgz + strdz * dz, egz - strdz * dz, dgz)
    xticks = np.arange(bgx, egx, dgx)
    ax.set_xticks(xticks)
    ax.set_yticks(zticks)
    ax.imshow(img,
              cmap=cmap,
              interpolation=interp,
              vmin=vmin,
              vmax=vmax,
              extent=[xmin, xmax, zmax, zmin],
              aspect=aspect)
    ax.grid(which='major', linestyle='-', color='g', linewidth=2)
    ax.set_ylabel(zlabel, fontsize=fsize)
    ax.set_xlabel(xlabel, fontsize=fsize)
    ax.tick_params(labelsize=fsize)
    tot3 = (nptchx) * (nptchz - 1)
    ax.set_title(
        r'Grid 3: X-stride=0, Z-stride=%d $\rightarrow \,\, %d\times%d = %d$ patches'
        % (strdz, nptchx, nptchz - 1, tot3),
        fontsize=fsize)
    if (pltcoords):
      plot_patchcoords(nptchz - 1, nptchx, zticks, xticks, zmax, xmax, 1, 0,
                       zshft, xshft, textsize, 'g')

  if (strdx != 0 and strdz != 0):
    fig = plt.figure(figsize=(kwargs.get('wbox', 10), kwargs.get('hbox', 6)))
    ax = fig.gca()
    zticks = np.arange(bgz + strdz * dz, egz - strdz * dz, dgz)
    xticks = np.arange(bgx + strdx * dx, egx - strdx * dx, dgx)
    ax.set_xticks(xticks)
    ax.set_yticks(zticks)
    ax.imshow(img,
              cmap=cmap,
              interpolation=interp,
              vmin=vmin,
              vmax=vmax,
              extent=[xmin, xmax, zmax, zmin],
              aspect=aspect)
    ax.grid(which='major', linestyle='-', color='b', linewidth=2)
    ax.set_ylabel(zlabel, fontsize=fsize)
    ax.set_xlabel(xlabel, fontsize=fsize)
    ax.tick_params(labelsize=fsize)
    tot4 = (nptchx - 1) * (nptchz - 1)
    ax.set_title(
        r'Grid 4: X-stride=%d, Z-stride=%d $\rightarrow \,\, %d\times%d = %d$ patches'
        % (strdx, strdz, nptchx - 1, nptchz - 1, tot4),
        fontsize=fsize)
    if (pltcoords):
      plot_patchcoords(
          nptchz - 1,
          nptchx - 1,
          zticks,
          xticks,
          zmax,
          xmax,
          1,
          1,
          zshft,
          xshft,
          textsize,
          'b',
      )

  print("Total number of patches: %d = %d + %d + %d + %d" %
        (totptchs, tot1, tot2, tot3, tot4))

  plt.show()


def plot_patchcoords(nptchz, nptchx, zticks, xticks, zmax, xmax, zidxi, xidxi,
                     zshft, xshft, textsize, color):
  """
  Plots the patch coordinates on the image
  To be used with plot_patchgrid2d
  """
  idx = zticks < zmax
  zticksw = zticks[idx]
  idx = xticks < xmax
  xticksw = xticks[idx]
  # Get the offset
  xtickdx = xticksw[1] - xticksw[0]
  ztickdx = zticksw[1] - zticksw[0]
  for iztick in range(nptchz):
    zpos = zticksw[iztick] + ztickdx / 2
    xidx = xidxi
    for ixtick in range(nptchx):
      xpos = xticksw[ixtick] + xtickdx / 2
      plt.text(xpos,
               zpos,
               '(%d,%d)' % (zidxi, xidx),
               fontsize=textsize,
               color=color)
      xidx += xshft
    zidxi += zshft


def patch_window2d(img, nzp, nxp, transp=False, pad=False) -> np.ndarray:
  """
  Windows the data based on the provided patch size so that
  the patches fit evenly into the image

  Parameters:
    img    - the input image [n3,nz,nx]
    nzp    - the size of the patch in z
    nxp    - the size of the patch in x
    transp - whether to transpose the input image [False]
    pad    - pad instead of window [False]
  """
  # Get the image shape
  if (transp):
    nx, nz = img.shape[-2:]
  else:
    nz, nx = img.shape[-2:]

  nzw = nz - divmod(nz, nzp)[1]
  nxw = nx - divmod(nx, nxp)[1]

  if (len(img.shape) == 2):
    return img[:nzw, :nxw]
  else:
    return img[..., :nzw, :nxw]
