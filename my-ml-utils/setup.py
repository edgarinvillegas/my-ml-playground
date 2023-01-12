#!/usr/bin/env python

from distutils.core import setup

setup(name='my_ml_utils_dist',  # name just for installation
      version='0.0',
      description="Edgarin's ML Utilities",
      author='Edgar Villegas',
      author_email='edgarinvillegas@hotmail.com',
      install_requires=['torch'],  # consumer dependency
      packages=['train_validate_test_package'], # names for importing
     )