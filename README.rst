===============================
edrixs
===============================

.. image:: https://img.shields.io/travis/mrakitin/edrixs.svg
        :target: https://travis-ci.org/mrakitin/edrixs

.. image:: https://img.shields.io/pypi/v/edrixs.svg
        :target: https://pypi.python.org/pypi/edrixs


An open source toolkit for simulating RIXS spectra based on exact diagonalization (ED) for strongly correlated materials.
It is developed as part of `COMSCOPE project <https://www.bnl.gov/comscope/software/comsuite.php/>`_ in the Center for Computational Material Spectroscopy and Design, Brookhaven National Laboratory

* Free software: GNU General Public License Version 3
* Documentation: https://nsls-ii.github.io/edrixs.

Features
--------

* ED solver
* XAS spectra
* RIXS spectra

Installation
------------
* Required tools and libraries

   * gfortran (recommended) or ifort 
   * MPI environment (openmpi or mpich)
   * mpif90 (binding with gfortran or ifort)
   * Python3
   * BLAS and LAPACK (gfortran+OpenBLAS or ifort+MKL)
   * arpack-ng (enable mpi)
   * Numpy
   * Scipy
   * Sympy
   * Matplotlib
   * mpi4py
   * Sphinx
   * Numpydoc

  Be sure to compile OpenBLAS, arpack-ng, mpi4py and edrixs with the same Fortran (MPI) compiler.

* Install Fortran parts of edrixs

    .. code-block:: bash

       $ cd src

  edit make.sys to set the correct libraries of BLAS/LAPACK, arpack-ng and f2py compiler options.

    .. code-block:: bash

       $ make
       $ make install

  There will be problems when using gfortran and f2py with MKL, so we recommend gfortran+OpenBLAS. libedrixsfortran.a will be generated, which will be used when building python interface. The executable .x files will be installed in bin directory and add the following line in .bashrc or .bash_profile file,

    .. code-block:: bash

       export PATH=edrixs/bin:$PATH

* Install Python parts of edrixs

  Be sure to first make libedrixsfortran.a in src.

    .. code-block:: bash

       $ python setup.py config_fc --f77exec=mpif90 --f90exec=mpif90 build_ext \
         --libraries=openblas,parpack,arpack --library-dirs=/usr/lib:/opt/local/lib \
         --link-objects=./src/libedrixsfortran.a
       $ pip install .

  where, **--library-dirs** ares the paths to search **--libraries**, please set it according to your environments.


How to cite
-----------
If you are using the EDRIXS code to do some studies and would like to publish your great works, it would be really appreciated if you can cite the following paper

 .. code-block:: bash

   EDRIXS: An open source toolkit for simulating spectra of resonant inelastic x-ray scattering
   Y.L. Wang, G. Fabbris, M.P.M. Dean and G. Kotliar, arXiv:1812.05735. Accepted as publication in CPC.

