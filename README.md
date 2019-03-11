# EDRIXS 

EDRIXS is an open source toolkit for the simulation of RIXS spectrum based on exact diagonalization for strongly correlated materials.
It is designed as a post-processing tool for the open source [COMSCOPE project](https://www.bnl.gov/comscope/software/comsuite.php) developed in the Center for
Computational Material Spectroscopy and Design in Brookhaven National Laboratory.

## License

* Free software: 3-clause BSD license

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
$ cd edrixs/src
$ cp make.sys.ifort make.sys (or cp make.sys.gfortran make.sys)
$ editor make.sys
$ make 
$ make install
```
where, **edrixs** is where the EDRIXS code is uncompressed. There will be problems when using gfortran with MKL, so we recommend ifort+MKL or gfortran+openblas. Be sure to compile arpack-ng with the same mpif90 compiler and BLAS/LAPACK libraries.

## How to cite

If you are using the EDRIXS code to do some studies and would like to publish your great works, it would be really appreciated if you can cite the following paper:

```sh
EDRIXS: An open source toolkit for simulating spectra of resonant inelastic x-ray scattering
Y.L. Wang, G. Fabbris, M.P.M. Dean and G. Kotliar, arXiv:1812.05735.
```
