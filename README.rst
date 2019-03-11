# EDRIXS 

EDRIXS is an open source toolkit for the simulation of RIXS spectrum based on exact diagonalization for strongly correlated materials.
It is designed as a post-processing tool for the open source [COMSCOPE project](https://www.bnl.gov/comscope/software/comsuite.php) developed in the Center for
Computational Material Spectroscopy and Design in Brookhaven National Laboratory.

## Version

v1.0.0 (devel)

## License

GNU General Public License Version 3

## Features

* ED solver
* XAS spectrum
* RIXS spectrum

## Installation

* Required tools and libraries
    * Intel Fortran (ifort) or GNU gfortran compilers
    * MPI (openmpi or mpich)
    * Python3
    * BLAS and LAPACK (ifort+MKL or gfortran+OpenBLAS)
    * arpack-ng
    * Numpy
    * Scipy
    * Sympy
    * Matplotlib
    * Sphinx
    * Numpydoc

* Make Fortran src
```sh
$ cd edrixs/src/fortran
$ cp make.sys.ifort make.sys (or cp make.sys.gfortran make.sys)
$ editor make.sys
$ make 
$ make install
```
where, **edrixs** is where the EDRIXS code is uncompressed. There will be problems when using gfortran with MKL, so we recommend ifort+MKL or gfortran+openblas. Be sure to compile arpack-ng with the same mpif90 compiler and BLAS/LAPACK libraries.

The executable .x files will be installed in bin directory. Add the following two lines in .bashrc or .bash_profile file
```sh
export PATH=edrixs/bin:$PATH
export PYTHONPATH=edrixs/src/python:$PYTHONPATH
```
* Make Python Documentation
```sh
$ cd edrixs/docs
$ mkdir build
$ sphinx-build -b html source build
$ make html
```
and open the file
```sh
edrixs/docs/build/index.html
```
in a browser to read the documentation.

## How to cite

If you are using the EDRIXS code to do some studies and would like to publish your great works, it would be really appreciated if you can cite the following paper:

```sh
EDRIXS: An open source toolkit for simulating spectra of resonant inelastic x-ray scattering
Y.L. Wang, G. Fabbris, M.P.M. Dean and G. Kotliar, arXiv:1812.05735.
```

## Contact
```sh
Yilin Wang
Department of Condensed Matter Physics and Materials Science, Brookhaven National Laboratory, Upton, New York 11973, USA
email: wangyilin2015@gmail.com or yilinwang@bnl.gov
```
