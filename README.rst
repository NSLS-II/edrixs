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
   * Intel Fortran (ifort) or GNU gfortran compiler
   * MPI environment (openmpi or mpich)
   * Python3
   * BLAS and LAPACK (ifort+MKL or gfortran+OpenBLAS)
   * arpack-ng
   * Numpy
   * Scipy
   * Sympy
   * Matplotlib
   * Sphinx
   * Numpydoc

* Install Python parts of edrixs
    .. code-block:: bash

       $ python setup.py install

* Install Fortran parts of edrixs
    .. code-block:: bash

       $ cd edrixs/src
       $ cp make.sys.ifort make.sys (or cp make.sys.gfortran make.sys)
       $ editor make.sys
       $ make
       $ make install

  where, edrixs is where the code is uncompressed. There will be problems when using gfortran with MKL, so we recommend ifort+MKL or gfortran+OpenBLAS. Be sure to compile arpack-ng with the same mpif90 compiler and BLAS/LAPACK libraries. 
  
  The executable .x files will be installed in bin directory. Add the following line in .bashrc or .bash_profile file,
    .. code-block:: bash

      export PATH=edrixs/bin:$PATH

How to cite
-----------
If you are using the EDRIXS code to do some studies and would like to publish your great works, it would be really appreciated if you can cite the following paper
 .. code-block:: bash

   EDRIXS: An open source toolkit for simulating spectra of resonant inelastic x-ray scattering
   Y.L. Wang, G. Fabbris, M.P.M. Dean and G. Kotliar, arXiv:1812.05735. Accepted as publication in CPC.

