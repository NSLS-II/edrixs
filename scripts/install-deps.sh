#!/bin/bash

set -vxeuo pipefail

platform="$(uname)"

if [ "${platform}" == "Linux" ]; then
    sudo apt-get update -y
    sudo apt-get install -y \
        gfortran \
        openmpi-bin \
        openmpi-doc \
        libopenmpi-dev \
        libopenblas-dev \
        libarpack2-dev \
        libparpack2-dev
elif [ "${platform}" == "Darwin" ]; then
    echo "TODO: fix OSX dependencies"
fi

# These packages are installed in the base environment but may be older
# versions. Explicitly upgrade them because they often create
# installation problems if out of date.
python -m pip install --upgrade pip setuptools wheel numpy

# # Generate .whl file.
python setup.py sdist bdist_wheel

# Install this package and the packages listed in requirements.txt.
pip install -v .

# Install extra requirements for running tests and building docs.
pip install -r requirements-dev.txt

# List the depencencies
pip list
