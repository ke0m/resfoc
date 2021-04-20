from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
import subprocess
import site

VERSION = '0.0.1'
DESCRIPTION = """A package for residual focusing
                 of seismic images"""
LONG_DESCRIPTION = """A package for residual focusing of
                     of seismic images. Has the capabilities
                     to perform the residual focusing using tradtional
                     signal processing techniques (semblance) as well as
                     using deep learning techniques. Also, contains code
                     for generating training images and training a seismic
                    fault interpretation network"""


class Build(build_py):
  """Custom build for building PyBind11 modules"""

  def run(self):
    # Get the external libs
    cmdsub = 'git submodule init && git submodule update'
    subprocess.check_call(cmdsub, shell=True)
    # Residual focusing
    cmdtway = "cd ./resfoc/resfoc/src && make INSTALL_DIR=%s" % (
        site.getsitepackages()[0])
    subprocess.check_call(cmdtway, shell=True)
    # Model building
    cmdoway = "cd ./resfoc/velocity/src && make INSTALL_DIR=%s" % (
        site.getsitepackages()[0])
    subprocess.check_call(cmdoway, shell=True)
    build_py.run(self)


class Develop(develop):
  """Custom build for building PyBind11 modules in development mode"""

  def run(self):
    # Get the external libs
    cmdsub = 'git submodule init && git submodule update'
    subprocess.check_call(cmdsub, shell=True)
    # Residual focusing
    cmdtway = "cd ./resfoc/resfoc/src && make INSTALL_DIR=%s" % (
        site.getsitepackages()[0])
    subprocess.check_call(cmdtway, shell=True)
    # Model building
    cmdoway = "cd ./resfoc/velocity/src && make INSTALL_DIR=%s" % (
        site.getsitepackages()[0])
    subprocess.check_call(cmdoway, shell=True)
    develop.run(self)


# Setting up
setup(
    name="resfoc",
    version=VERSION,
    author="Joseph Jennings",
    author_email="<joseph29@sep.stanford.edu>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    cmdclass={
        'build_py': Build,
        'develop': Develop,
    },
    keywords=['seismic', 'imaging', 'interpretation', 'focusing'],
    classifiers=[
        "Intended Audience :: Seismic processing/imaging",
        "Programming Language :: Python :: 3",
        "Operating System :: Linux ",
    ],
)
