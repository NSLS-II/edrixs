===============================
edrixs
===============================

.. image:: https://img.shields.io/travis/NSLS-II/edrixs.svg
        :target: https://travis-ci.org/NSLS-II/edrixs

.. image:: https://img.shields.io/pypi/v/edrixs.svg
        :target: https://pypi.python.org/pypi/edrixs


An open source toolkit for simulating RIXS spectra based on exact diagonalization (ED) for strongly correlated materials.
`It is developed <https://www.bnl.gov/comscope/software/EDRIXS.php>`_ as part of `COMSCOPE project <https://www.bnl.gov/comscope/software/comsuite.php/>`_ in the Center for Computational Material Spectroscopy and Design, Brookhaven National Laboratory

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

   * Fortran compiler: gfortran and ifort are supported 
   * MPI environment: openmpi and mpich are tested
   * mpif90 (binding with gfortran or ifort) and mpicc (binding with gcc)
   * Python3
   * BLAS and LAPACK: gfortran+OpenBLAS or ifort+MKL
   * arpack-ng (with mpi enabled)
   * Numpy
   * Scipy
   * Sympy
   * Matplotlib
   * mpi4py
   * Sphinx
   * Numpydoc

  Be sure to compile OpenBLAS, arpack-ng, mpi4py and edrixs with the same (MPI) Fortran compiler.

* Install Fortran parts of edrixs

    .. code-block:: bash

       $ cd src
       $ make F90=mpif90 LIBS="-L/usr/local/lib -lopenblas -lparpack -larpack"
       $ make install

  where, you may need to change ``F90`` and ``LIBS`` according to your specific environment. There will be problems when using gfortran with MKL, so we recommend ``gfortran+OpenBLAS`` or ``ifort+MKL``. ``libedrixsfortran.a`` will be generated, which will be used when building python interface. The executable ``.x`` files will be installed in ``edrixs/bin`` directory and add the following line in ``.bashrc`` or ``.bash_profile`` file,

    .. code-block:: bash

       export PATH=/root_dir_of_edrixs/edrixs/bin:$PATH

* Install Python parts of edrixs

  Be sure to first make ``libedrixsfortran.a`` in src.

    .. code-block:: bash

       $ python setup.py config_fc --f77exec=mpif90 --f90exec=mpif90 build_ext \
         --libraries=openblas,parpack,arpack --library-dirs=/usr/lib:/usr/local/lib:/opt/local/lib \
         --link-objects=./src/libedrixsfortran.a
       $ pip install .

  where, ``--library-dirs`` ares the paths to search ``--libraries``, please set it according to your environments.


Run edrixs in docker
--------------------
To make life easier, we have built a docker image based on Ubuntu Linux (18.04) for edrixs, so you don't need to struggle with the installation anymore. 
The docker image can be used on any OS as long as the `docker <https://www.docker.com/>`_ application are available.
Follow these steps to use the docker image:

* Install the `docker <https://www.docker.com/>`_ application on your system and `learn how to use it <https://docs.docker.com/get-started/>`_.
* Once the docker is running, create a directory to store data in your host OS and launch a container to run edrixs

    .. code-block:: bash
      
       $ mkdir /dir/on/your/host/os   # A directory on your host OS
       $ docker pull edrixs/edrixs    # pull latest version
       $ docker run -it -p 8888:8888 -u rixs -w /home/rixs -v /dir/on/your/host/os:/home/rixs/data edrixs/edrixs
       
  it will take a while to pull the image from `Docker Hub <https://cloud.docker.com/repository/docker/edrixs/edrixs/>`_ for the first time, while, it will launch the local one very fast at the next time.
  
  * ``-p 8888:8888`` maps container's port 8888 to host port 8888.
  * ``-u rix`` means using a default user ``rixs`` to login the Ubuntu Linux, the password is ``rixs``. 
  * ``-v /dir/on/your/host/os:/home/rixs/dat`` means mounting the directory ``/dir/on/your/host/os`` from your host OS to  ``/home/rixs/data`` on the Ubuntu Linux in the container. 
   
* After launching the container, you will see ``data`` and ``edrixs_examples`` in ``/home/rixs`` directory. If you want to save the data from edrixs calculations to your host system, you need to work in ``/home/rixs/data`` directory.

    .. code-block:: bash
    
       $ cd /home/rixs/data
       $ cp -r ../edrixs_examples .
       $ Play with edrixs ... 

  Note that any changes outside ``/home/rixs/data`` will be lost when this container stops. You can only use your host OS to make interactive plots. Use ``sudo apt-get install`` to install softwares if they are needed. 
  
* Type ``exit`` in the container to exit. You can delete all the stopped containers by

   .. code-block:: bash
      
      $ docker rm $(docker ps -a -q)

* You can delete the edrixs image by

   .. code-block:: bash
   
      $ docker rmi edrixs/edrixs   


How to cite
-----------
If you are using the EDRIXS code to do some studies and would like to publish your great works, it would be really appreciated if you can cite the following paper

 .. code-block:: bash

   EDRIXS: An open source toolkit for simulating spectra of resonant inelastic x-ray scattering
   Y.L. Wang, G. Fabbris, M.P.M. Dean and G. Kotliar, arXiv:1812.05735. Accepted as publication in CPC.

