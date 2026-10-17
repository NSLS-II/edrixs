.. _backend:

*******
Backend
*******

The :ref:`introduction` describes the four stages of a calculation.  Stages two
to four -- building operators, diagonalizing, and assembling spectra --
delegate their numerical work to a backend selected with ``backend`` when
calling ``build_op`` or ``get_ops``:

* ``'dense'`` builds full NumPy matrices and uses exact dense
  diagonalization and explicit eigenstate sums.  It is intended for small
  problems, reference calculations, and pedagogy.
* ``'scipy'`` builds SciPy CSR matrices and uses LOBPCG, Lanczos, and GMRES.
  It is the default backend.
* ``'fortran'`` uses the native MPI solvers and communicates through files in
  the current working directory.  We provide the function
  :func:`edrixs.get_ops_disk <edrixs.solvers.get_ops_disk>` to load the setup
  from disk if desired.
* ``'petsc'`` uses distributed PETSc matrices, SLEPc eigensolvers, and PETSc
  Krylov solvers.  It is intended for larger Hilbert spaces and MPI runs.

Backend-specific numerical controls belong in a ``backend_kws`` mapping. The
mapping is specific to both the backend *and the operation*.

.. _lazy-operators:

Operator factories and solver dispatch
======================================

``build_op`` returns a zero-argument ``functools.partial`` with a ``.backend``
attribute. Numerical ``get_ops`` returns these factories for both Hamiltonians
and every transition component. Public ``ed``, ``xas``, and ``rixs`` read the
labels and build each distinct factory once per invocation; they no longer
accept a ``backend`` keyword. All supplied Hamiltonians and transition components
must have matching labels. Missing labels raise ``TypeError``; unknown or mixed
backend labels raise ``ValueError``, before any matrix is constructed.
Fortran disk handles also carry a label and pass directly to the Fortran solvers.

For example::

    H = edrixs.build_op(emat, umat, basis, backend='scipy')
    print(H.backend)              # 'scipy'; no matrix built yet
    energies, states = edrixs.ed(H, num_evals=2)
    matrix = H()                 # Explicit construction for matrix arithmetic

The spectroscopy workflow passes the factories directly to each solver::

    *problem, shift = edrixs.model_1v1c(('p', 's'), shell_level=(0, -100))
    hmat_i, hmat_n, trans_ops = edrixs.get_ops(*problem, backend='scipy')
    eval_i, evec_i = edrixs.ed(hmat_i, shift=shift, num_evals=2)
    absorption = edrixs.xas(eval_i, evec_i, hmat_n, trans_ops, ominc)
    scattering = edrixs.rixs(
        eval_i, evec_i, hmat_i, hmat_n, trans_ops, ominc, eloss,
    )

Here ``ominc`` and ``eloss`` are the incident-energy and energy-loss grids.
Selecting ``backend`` in ``get_ops`` also selects the backend for all three
solver calls. ``trans_ops`` is a sequence of factories, one per Cartesian
transition component.

Each explicit factory call builds a fresh matrix. There is no cross-call cache.
Calling ``ed(H)`` and then ``rixs(..., H, ...)`` builds ``H`` separately for each
solver call. Within a single call, repeated references to the same factory
share one constructed matrix.
Coefficient arrays are retained by reference, so changes to them affect later
builds. Basis preparation occurs when creating the factory; backend matrix
construction and its option validation occur when calling it, either explicitly
or inside a solver.

Migrating existing calls
-------------------------

Keep ``backend=`` on ``build_op`` and ``get_ops``. Remove it from ``ed``, ``xas``,
and ``rixs``, and pass the labeled factory directly. Retain any ``backend_kws``
that configure the solver.

For matrix arithmetic or direct access to matrix methods, call the factory
explicitly::

    number_op = edrixs.build_op(number_coefficients, None, basis, backend='dense')
    number_matrix = number_op()
    expectation = state.conj() @ (number_matrix @ state)

The resulting raw matrix does not carry the factory's label. Public solvers
require labeled factories or Fortran handles; code that already owns concrete
matrices can use backend-specific functions directly::

    from edrixs.scipy_backend.scipy_backend import ed_scipy

    energies, states = ed_scipy(matrix, num_evals=2)

The same distinction applies to ``xas_scipy`` and ``rixs_scipy`` and the dense
and PETSc backend functions.

Numerical precision
===================

XAS and RIXS thermally average over the retained initial eigenstates.  Request
enough eigenpairs from ``ed`` to keep every thermally populated state in the
ground-state manifold, including complete degenerate or nearly degenerate
multiplets; otherwise the Boltzmann average is incomplete.

EDRIXS uses established numerical linear-algebra packages, but floating-point
and iterative solvers are not exact.  If numerical accuracy is a concern,
repeat the calculation with varied solver settings and check that the result
is unchanged.  Relevant settings include ``tol`` or ``eigval_tol``,
``maxiter``, ``blocksize`` or ``ncv``, ``nkryl``, and the RIXS ``linsys_*``
options documented below.

.. _backend-options:

Backend keyword reference
=========================

Choose the backend whose keywords you need:

* :doc:`SciPy keywords <backends/scipy>`
* :doc:`Dense keywords <backends/dense>`
* :doc:`Fortran keywords <backends/fortran>`
* :doc:`PETSc keywords <backends/petsc>`

These references apply to the staged ``get_ops`` / ``ed`` / ``xas`` / ``rixs``
interface.  The legacy ``ed_*_fort``, ``xas_*_fort``, and ``rixs_*_fort``
functions retain their existing explicit parameters.

.. toctree::
   :hidden:

   backends/scipy
   backends/dense
   backends/fortran
   backends/petsc
