#!/usr/bin/env python
###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: MIT
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

import os
import imp
import sys
from distutils.core import setup

import numpy

minimum_numpy_version = "1.6"

# check versions
if sys.version_info < (2, 6):
    raise RuntimeError("must use python 2.6 or greater")

if numpy.__version__ < minimum_numpy_version:
    print("*Error*: NumPy version is lower than needed: %s < %s" %
          (numpy.__version__, minimum_numpy_version))
    sys.exit(1)

    
def get_version(filepath):
    dirpath = os.path.dirname(os.path.abspath(__file__))
    fullpath = os.path.join(dirpath, filepath)
    version_module = imp.load_module("__version_module__", open(fullpath),
                                     fullpath, ('.py', 'U', 1))
    return version_module.version

setup(name='numexpr-numba',
      version=get_version('numexpr/version.py'),
      description='Fast numerical expression evaluator for NumPy, based on Numba',
      author='Gaetan de Menten, David M. Cooke, Francesc Alted and others',
      author_email='gdementen@gmail.com, david.m.cooke@gmail.com, faltet@pytables.org',
      url='http://code.google.com/p/numexpr/',
      license = 'MIT',
      packages = ['numexpr']
)