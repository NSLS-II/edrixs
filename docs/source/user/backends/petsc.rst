.. _petsc-backend-options:

**************
PETSc keywords
**************

:doc:`Back to the backend overview <../backend>`

About this backend
==================

PETSc supplies distributed vectors and matrices together with scalable linear
solvers and preconditioners; its concise
`architecture overview <https://petsc.org/release/overview/nutshell/>`__
introduces the ``Vec``, ``Mat``, ``KSP``, and ``PC`` objects.  Python programs
access them through the official
`petsc4py API <https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.html>`__.
For eigenvalue problems, SLEPc builds on PETSc and exposes its ``EPS`` solver
through the
`SLEPc eigenvalue guide <https://slepc.upv.es/release/documentation/manual/eps.html>`__
and the
`slepc4py EPS API <https://slepc.upv.es/release/slepc4py/reference/slepc4py.SLEPc.EPS.html>`__.

EDRIXS constructs distributed PETSc ``Mat`` objects, obtains low-energy
eigenpairs with SLEPc's Krylov--Schur method, and evaluates XAS and RIXS with
distributed Lanczos and KSP solves.  Importing EDRIXS does not import PETSc;
``petsc4py`` and ``slepc4py`` are loaded only when this backend is used.

``build_op`` and ``get_ops`` return factories labeled with
``.backend == 'petsc'``. Calling a factory assembles its distributed matrix;
public ``ed``, ``xas``, and ``rixs`` read the labels and pass the factories to
the PETSc backend. All MPI ranks in the matrix communicator must participate
in construction and solver calls. See :ref:`lazy-operators`.

Options are specific to the stage where they are supplied.

Matrix lifetime
===============

Only PETSc ``rixs`` accepts ``matrix_lifetime`` in ``backend_kws``:

* ``'retain'`` (default) builds each required factory on first use, keeps its
  matrix for reuse during the calculation, and destroys it before returning.
* ``'recompute'`` builds a matrix for each operation and destroys it immediately
  afterward, rebuilding it for later uses. Each linear solve or
  Lanczos run keeps its Hamiltonian for that complete phase. Transition
  components are built and released individually during each application.

For example::

    spectrum = edrixs.rixs(
        eval_i, evec_i, hmat_i, hmat_n, trans_ops, ominc, eloss,
        backend_kws={'matrix_lifetime': 'recompute', 'nkryl': 200},
    )

``'recompute'`` keeps at most one assembled many-body operator resident at
a time, at the cost of repeated assembly.
It releases the intermediate Hamiltonian before emission and final-state
Lanczos calculations, while preserving shared correction vectors across
outgoing polarization channels. Unused operators are not constructed.

Factories passed to PETSc must return fresh matrices that the backend can
modify and destroy. Backend-specific solvers also require factories; repeated
arguments may be instantiated independently. Returned eigenvectors remain owned
by the caller. Matrices obtained by explicitly calling a factory outside a solver
are also owned by the caller. All ranks must use the same lifetime policy.

``build_op`` and ``get_ops``
============================

.. backend-options:: petsc get_ops

``nnz_guess_per_row`` controls PETSc matrix preallocation.  Leaving it unset
uses an estimate from the retained one- and two-body coefficients. With multiple
MPI ranks, the hint is applied separately to the local-column and remote-column
blocks of each rank's matrix rows. With one rank it applies to the sequential
matrix. Larger hints reserve more memory but can avoid reallocations during
assembly; they do not change the matrix values. A
``mat_type`` such as ``'aijcusparse'`` can select an accelerated matrix format;
see PETSc's `matrix type overview
<https://petsc.org/release/manual/mat/#basic-matrix-operations>`__.

``ed``
======

Pass the final model output as ``ed(hmat_i, shift=shift)`` to subtract the
filled-core shell energy from the factory-created distributed Hamiltonian before
diagonalization. Returned eigenvalues retain the original energy reference.
With a nonzero ``shift``, ``eigval_tol`` is an absolute residual tolerance for
the shifted problem, avoiding division by a near-zero centered eigenvalue.
With ``shift=0``, it is a relative residual tolerance, scaled by the eigenvalue
magnitude.

ED calls its factory once and destroys the matrix before returning.
``matrix_lifetime`` is not accepted for ED.

.. backend-options:: petsc ed

The eigensolver uses SLEPc ``EPS`` with a Hermitian problem type and requests
the smallest-real eigenpairs.  ``ncv`` sets the Krylov subspace size described
by `EPS.setDimensions
<https://slepc.upv.es/release/slepc4py/reference/slepc4py.SLEPc.EPS.html#slepc4py.SLEPc.EPS.setDimensions>`__.

``xas``
=======

XAS builds all supplied factories once at calculation start and retains the
matrices for the calculation. Factory-created matrices are destroyed before
returning. ``matrix_lifetime`` is not accepted for XAS.

XAS and RIXS center factory-created distributed matrices using the lowest retained initial
energy and the first incident energy, which should lie near resonance. RIXS
also centers the final Hamiltonian. Spectra are evaluated on these relative
energies before restoring the original reference in returned pole dictionaries.
Factory-created Hamiltonians are shifted in place.

.. backend-options:: petsc xas

``rixs``
========

.. backend-options:: petsc rixs

The default solver is ``shifted_pminres``. ``linsys_tol`` sets KSP's absolute
tolerance, ``atol``, and the default relative tolerance is zero. Other KSP
types retain PETSc's default relative tolerance unless overridden through
its options database.

EDRIXS calls KSP's ``setFromOptions()`` after applying ``backend_kws``.
Corresponding PETSc options-database entries, such as ``ksp_atol``, ``ksp_rtol``,
``ksp_max_it``, and ``ksp_type``, take precedence over these settings.

``ksp_type`` accepts a PETSc KSP type such as ``'gmres'``; the available
algorithms and their tradeoffs are listed in the `KSP solver table
<https://petsc.org/release/manual/ksp/#tab-kspdefaults>`__.

To select GMRES instead of the default shifted PMINRES, pass::

    backend_kws={'ksp_type': 'gmres', 'linsys_tol': 1e-10}
