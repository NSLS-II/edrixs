#!/bin/bash

set -vxeuo pipefail
sudo apt-get install -y \
    gfortran \
    openmpi-bin \
    openmpi-doc \
    libopenmpi-dev \
    libopenblas-dev \
    libarpack2-dev \
    libparpack2-dev

# These packages are installed in the base environment but may be older
# versions. Explicitly upgrade them because they often create
# installation problems if out of date.
python -m pip install --upgrade pip "setuptools<=65.5.*" numpy

# Install this package and the packages listed in requirements.txt.
pip install -v .

# Generate .whl file.
python setup.py bdist_wheel

# Install extra requirements for running tests and building docs.
pip install -r requirements-dev.txt

# List the depencencies
pip list
