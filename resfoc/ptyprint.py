import numpy as np
import sys

def create_inttag(numin,totnum):
  """ Creates a tag that is appended with zeros for friendly Unix sorting """
  nzeroso = int(np.log10(totnum)); nzeros = nzeroso
  tagout = None
  for izro in range(1,nzeroso+1):
    if((numin >= 10**(izro-1) and numin < 10**(izro))):
      tagout = '0'*(nzeros) + str(numin)
    nzeros -= 1
  if(tagout != None):
    return tagout
  elif(numin == 0):
    return '0'*(nzeroso) + str(numin)
  else:
    return str(numin)

def update_progress(progress):
  """
  update_progress() : Displays or updates a console progress bar

  Accepts a float between 0 and 1. Any int will be converted to a float.
  A value under 0 represents a 'halt'.
  A value at 1 or bigger represents 100%

  @author: Brian Khuu
  @source: https://stackoverflow.com/questions/3160699/python-progress-bar
  """
  barLength = 40 # Modify this to change the length of the progress bar
  status = ""
  if isinstance(progress, int):
    progress = float(progress)
  if not isinstance(progress, float):
    progress = 0
    status = "error: progress var must be float\r\n"
  if progress < 0:
    progress = 0
    status = "Halt...\r\n"
  if progress >= 1:
    progress = 1
    status = "Done...\r\n"
  block = int(round(barLength*progress))
  text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100,2), status)
  sys.stdout.write(text)
  sys.stdout.flush()

def printprogress(prefix,j,count,size=40,file=sys.stdout):
  """
  A modified version of progressbar

  @author: Joseph Jennings
  """
  x = int(size*j/count)
  file.write("%s[%s%s] %i/%i\r" % (prefix+" ", "#"*x, "."*(size-x), j, count))
  if(j == count):
    file.write("\n")
  file.flush()

def progressbar(it, prefix="", size=40, file=sys.stdout, verb=True):
  """
  Progress bar

  @author: eusoubrasileiro
  @source: https://stackoverflow.com/questions/3160699/python-progress-bar
  """
  count = len(it)
  if(count == 0): count += 1
  def show(j):
    x = int(size*j/count)
    file.write("%s[%s%s] %i/%i\r" % (prefix+" ", "#"*x, "."*(size-x), j, count))
    file.flush()
  if(verb): show(0)
  for i, item in enumerate(it):
    yield item
    if(verb): show(i+1)
  if(verb):
    file.write("\n")
    file.flush()

