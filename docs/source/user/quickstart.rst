===================
Quickstart tutorial
===================

Use edrixs as an ED calculator
------------------------------
edrixs can be used as a simple ED calculator to get eigenvalues (eigenvectors) of a many-body Hamiltonian with small size dimension (:math:`< 1,000`).
We will give an example to get eigenvalues for a :math:`p`-orbital system (:math:`l=1`). There are 6 orbitals including spin.

Launch your favorite python terminal::
    
    >>> import edrixs
    >>> norb = 6    # number of orbitals
    >>> U, J = 4.0, 1.0    # Coulomb U and Hund's coupling J
    >>> mu = 5 * (U / 2 - J)    # chemical potential, particle-hole symmetry
    >>> emat = - np.eye(norb) * mu
    >>> umat = edrixs.get_umat_kanamori(norb, U, J)    # Kanamori type of Coulomb interaction
    >>> eigval = []
    >>> eigvec = []
    >>> for noccu in range(norb+1):    # loop over all the possible occupancy
    ...     basis = edrixs.get_fock_bin_by_N(norb, noccu)    # Fock basis
    ...     H = edrixs.two_fermion(emat, basis, basis)    # Hamiltonian, two fermion terms
    ...     H += edrixs.four_fermion(umat, basis)    # Hamiltonian, four fermion terms
    ...     e, v = scipy.linalg.eigh(H)  # do ED
    ...     print(noccu, e)
    ...     eigval.append(e)
    ...     eigvec.append(v)
    0 [0.]
    1 [-5. -5. -5. -5. -5. -5.]
    2 [-9. -9. -9. -9. -9. -9. -9. -9. -9. -7. -7. -7. -7. -7. -4.]
    3 [-12. -12. -12. -12.  -9.  -9.  -9.  -9.  -9.  -9.  -9.  -9.  -9.  -9.  -7.  -7.  -7.  -7.  -7.  -7.]
    4 [-9. -9. -9. -9. -9. -9. -9. -9. -9. -7. -7. -7. -7. -7. -4.]
    5 [-5. -5. -5. -5. -5. -5.]
    6 [0.]

Hello RIXS!
-----------

This is a "Hello World!" example for RIXS simulations at Ni (:math:`3d^8`) :math:`L_{2/3}` edges.
:math:`L_3` means transition from Ni-:math:`2p_{3/2}` to Ni-:math:`3d`, and
:math:`L_2` means transition from Ni-:math:`2p_{1/2}` to Ni-:math:`3d`.

.. plot:: pyplots/helloworld.py
   :include-source:
