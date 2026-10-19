"""PETSc-specific many-body operator construction helpers.

Fock-basis representations and ranking live in :mod:`edrixs.fock_basis`.
This module owns PETSc work decomposition, preallocation, insertion, and final
matrix assembly.  The shared operator code is used only as a bounded-range
matrix-entry kernel.
"""

from __future__ import annotations

import numpy as np

from .._operator_builder import prepare_operator_entry_kernel
from ..fock_basis import FockBinByN, get_fock_basis_combinadic

__all__ = [
    'FockBinByN', 'get_fock_basis_petsc', 'build_op_petsc_matrix',
]


# Backwards-compatible name from the initial PETSc implementation.
get_fock_basis_petsc = get_fock_basis_combinadic


def _default_preallocation(emat, umat, nr, tol_e, tol_u):
    """Estimate row-wise preallocation from retained coefficient counts."""
    ne = int(np.count_nonzero(np.abs(emat) > tol_e)) if emat is not None else 0
    if umat is None:
        nu = 0
    elif hasattr(umat, 'data') and hasattr(umat, 'tocoo'):
        nu = int(np.count_nonzero(np.abs(umat.data) > tol_u))
    else:
        nu = int(np.count_nonzero(np.abs(umat) > tol_u))
    return max(8, min(nr, 16 + ne + 2 * min(nu, 32)))


def _create_matrix(PETSc, comm, nl, nr, preallocation):
    """Create and preallocate an AIJ matrix for row-wise insertion."""
    hmat = PETSc.Mat().create(comm=comm)
    hmat.setSizes(((None, nl), (None, nr)))
    hmat.setType(PETSc.Mat.Type.AIJ)
    if hmat.getComm().getSize() > 1:
        # MPI AIJ stores local and remote columns in separate sparse blocks.
        # A scalar hint leaves the remote block at PETSc's small default.
        preallocation = (preallocation, preallocation)
    hmat.setPreallocationNNZ(preallocation)
    hmat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    hmat.setUp()
    return hmat


def _work_ranges(hmat, chunk_cols):
    """Yield bounded ranges of locally owned basis columns."""
    cstart, cend = hmat.getOwnershipRangeColumn()
    for start in range(cstart, cend, chunk_cols):
        yield start, min(start + chunk_cols, cend)


def _insert_entries(hmat, rows, cols, vals, PETSc):
    """Insert one bounded contribution buffer using PETSc bulk row insertion."""
    if rows.size == 0:
        return

    rows = np.asarray(rows, dtype=PETSc.IntType)
    cols = np.asarray(cols, dtype=PETSc.IntType)
    vals = np.asarray(vals, dtype=PETSc.ScalarType)

    # Group entries by row for setValuesIJV. Sorting is bounded by one work
    # chunk; COO assembly does not use this insertion helper.
    order = np.argsort(rows, kind='mergesort')
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]

    # Keep row grouping and insertion out of Python's per-entry/per-row loops.
    # rowmap includes global rows, including remote rows generated from locally
    # owned columns. Repeated coordinates still accumulate with ADD_VALUES.
    indptr = np.concatenate((
        [0], np.flatnonzero(rows[1:] != rows[:-1]) + 1, [rows.size],
    )).astype(PETSc.IntType, copy=False)
    hmat.setValuesIJV(
        indptr, cols, vals, rowmap=rows[indptr[:-1]],
        addv=PETSc.InsertMode.ADD_VALUES,
    )


def _finalize_matrix(hmat, mat_type):
    """Assemble the matrix and apply an optional matrix-type conversion."""
    hmat.assemblyBegin()
    hmat.assemblyEnd()
    if mat_type is not None:
        hmat = hmat.convert(mat_type)
    return hmat


def build_op_petsc_matrix(
    emat,
    umat,
    lb,
    rb=None,
    *,
    comm,
    tol_e=1e-10,
    tol_u=1e-10,
    nnz_guess_per_row=None,
    mat_type=None,
    assembly_chunk_cols=4096,
    assembly_method='row',
    use_numba=True,
):
    """Build an AIJ matrix with chunked row insertion or bulk COO assembly."""
    from petsc4py import PETSc

    if rb is None:
        rb = lb
    if lb.norbs != rb.norbs:
        raise ValueError("left and right Fock bases must have the same norbs")

    assembly_chunk_cols = int(assembly_chunk_cols)
    if assembly_chunk_cols < 1:
        raise ValueError("assembly_chunk_cols must be a positive integer")

    nl, nr = len(lb), len(rb)
    if assembly_method == 'coo':
        hmat = PETSc.Mat().create(comm=comm)
        hmat.setSizes(((None, nl), (None, nr)))
        hmat.setType(PETSc.Mat.Type.AIJ)
        build_range = prepare_operator_entry_kernel(
            emat, umat, lb, rb, tol_e=tol_e, tol_u=tol_u, use_numba=use_numba,
        )

        # Count first so filling never needs concatenation or oversized buffers.
        nentries = 0
        for start, end in _work_ranges(hmat, assembly_chunk_cols):
            entries = build_range(start, end)
            nentries += entries[2].size
            del entries
        coo_rows = np.empty(nentries, dtype=PETSc.IntType)
        coo_cols = np.empty(nentries, dtype=PETSc.IntType)
        coo_vals = np.empty(nentries, dtype=PETSc.ScalarType)
        offset = 0
        for start, end in _work_ranges(hmat, assembly_chunk_cols):
            entries = build_range(start, end)
            stop = offset + entries[2].size
            coo_rows[offset:stop] = entries[0]
            coo_cols[offset:stop] = entries[1]
            coo_vals[offset:stop] = entries[2]
            offset = stop
            del entries

        # PETSc handles sorting, repeated coordinates, and remote contributions.
        hmat.setPreallocationCOO(coo_rows, coo_cols)
        del coo_rows, coo_cols
        hmat.setValuesCOO(coo_vals, addv=PETSc.InsertMode.INSERT_VALUES)
        del coo_vals
        if mat_type is not None:
            hmat = hmat.convert(mat_type)
        return hmat

    if nnz_guess_per_row is None:
        nnz_guess_per_row = _default_preallocation(
            emat, umat, nr, tol_e, tol_u
        )

    # PETSc owns the matrix before any many-body matrix entries are generated.
    hmat = _create_matrix(
        PETSc, comm, nl, nr, nnz_guess_per_row
    )

    build_range = prepare_operator_entry_kernel(
        emat,
        umat,
        lb,
        rb,
        tol_e=tol_e,
        tol_u=tol_u,
        use_numba=use_numba,
    )

    for start, end in _work_ranges(hmat, assembly_chunk_cols):
        rows, cols, vals = build_range(start, end)
        _insert_entries(hmat, rows, cols, vals, PETSc)

    return _finalize_matrix(hmat, mat_type)
