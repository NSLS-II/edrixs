===================
Quickstart tutorial
===================

Use edrixs as an ED calculator
------------------------------
edrixs can be used as a simple ED calculator to get eigenvalues (eigenvectors) of a many-body Hamiltonian with small size dimension (:math:`< 1,000`).
We will give an example to get eigenvalues for a :math:`p`-orbital system (:math:`l=1`). There are 6 orbitals (:math:`2(2l+1)`) including spin.

First, launch your favorite python terminal, ipython, for example::
    
    $ ipython
    >>> import edrixs
    >>> norb = 6
    >>> U, J = 4.0, 1.0
    >>> umat = edrixs.get_umat_kanamori(norb, U, J)

Hello RIXS!
-----------

This is a "Hello World!" example for RIXS simulations at Ni (:math:`3d^8`) :math:`L_{2/3}` edges.
:math:`L_3` means transition from Ni-:math:`2p_{3/2}` to Ni-:math:`3d`, and
:math:`L_2` means transition from Ni-:math:`2p_{1/2}` to Ni-:math:`3d`.

.. plot:: pyplots/helloworld.py
   :include-source:
