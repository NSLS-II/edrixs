"""COO assembly matches existing operators, including remote contributions."""

import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.sparse as sp

from edrixs.fock_basis import FockBasisSpec
from edrixs.solvers import build_op, ed, get_ops, rixs, xas


PETSc = pytest.importorskip('petsc4py.PETSc')


@pytest.mark.parametrize('basis_method', ['combinadic', 'explicit'])
@pytest.mark.parametrize('use_numba', [False, True])
@pytest.mark.parametrize('chunk_cols', [1, 3])
@pytest.mark.parametrize('kind', ['square', 'rectangular', 'zero', 'sparse_u'])
def test_coo_matches_scipy_and_row_assembly(basis_method, use_numba, chunk_cols, kind):
    if use_numba:
        pytest.importorskip('numba')
    rng = np.random.default_rng(1231)
    emat = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    umat = rng.normal(size=(4, 4, 4, 4)) + 1j * rng.normal(size=(4, 4, 4, 4))
    right = FockBasisSpec.from_args(2, 1, 2, 1)
    left = FockBasisSpec.from_args(2, 2, 2, 0) if kind == 'rectangular' else right
    if kind == 'zero':
        emat[:] = 0
        umat[:] = 0
    if kind == 'sparse_u':
        umat = sp.csr_matrix(umat.reshape(16, 16))
    common = dict(basis_method=basis_method, use_numba=use_numba)
    expected = build_op(emat, umat, left, right, backend='scipy', **common)().toarray()
    for method in ['row', 'coo']:
        matrix = build_op(
            emat, umat, left, right, backend='petsc', **common,
            backend_kws={'assembly_method': method, 'assembly_chunk_cols': chunk_cols},
        )()
        try:
            lo, hi = matrix.getOwnershipRange()
            columns = np.arange(expected.shape[1], dtype=PETSc.IntType)
            local = (matrix.getValues(np.arange(lo, hi, dtype=PETSc.IntType), columns)
                     if hi > lo else np.empty((0, len(columns)), dtype=complex))
            actual = np.concatenate(matrix.getComm().tompi4py().allgather(local))
            assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        finally:
            matrix.destroy()


def test_coo_sums_duplicate_diagonal_contributions():
    # Each two-electron diagonal receives two contributions of opposite sign.
    matrix = build_op(
        np.diag([1., -1.]), None, FockBasisSpec.from_args(2, 2), backend='petsc',
        use_numba=False, backend_kws={'assembly_method': 'coo'},
    )()
    try:
        vector = matrix.createVecRight()
        result = matrix.createVecLeft()
        try:
            vector.set(1)
            matrix.mult(vector, result)
            assert result.norm() == 0
        finally:
            result.destroy()
            vector.destroy()
    finally:
        matrix.destroy()


@pytest.mark.parametrize('lifetime', ['retain', 'recompute'])
def test_coo_spectra_match_dense_reference(small_1v1c_problem, lifetime):
    dense_ops = get_ops(*small_1v1c_problem[:7], backend='dense', use_numba=False)
    petsc_ops = get_ops(
        *small_1v1c_problem[:7], backend='petsc', use_numba=False,
        backend_kws={'assembly_method': 'coo', 'assembly_chunk_cols': 2},
    )
    energies, vectors = ed(dense_ops[0])
    state = PETSc.Vec().createMPI(vectors.shape[0], comm=PETSc.COMM_WORLD)
    lo, hi = state.getOwnershipRange()
    state.getArray()[:] = vectors[lo:hi, 0]
    incident, losses = [4., 4.3], [-.1, .1, .3]
    channels = [('linear', .2, 'left', 0), ('linear', .2, 'right', 0)]
    try:
        expected_xas = xas(energies, vectors, dense_ops[1], dense_ops[2], incident)
        actual_xas = xas(energies, [state], petsc_ops[1], petsc_ops[2], incident)
        assert_allclose(actual_xas, expected_xas, rtol=1e-9, atol=1e-10)
        expected_rixs = rixs(energies, vectors, *dense_ops, incident, losses, pol_type=channels)
        actual_rixs = rixs(
            energies, [state], *petsc_ops, incident, losses, pol_type=channels,
            backend_kws={'matrix_lifetime': lifetime, 'linsys_tol': 1e-12},
        )
        assert_allclose(actual_rixs, expected_rixs, rtol=1e-8, atol=1e-9)
    finally:
        state.destroy()


def test_coo_preserves_requested_matrix_conversion():
    expected = np.array([[1., .2j], [-.2j, 2.]])
    matrix = build_op(
        expected, None, FockBasisSpec.from_args(2, 1), backend='petsc', use_numba=False,
        backend_kws={'assembly_method': 'coo', 'mat_type': 'dense', 'comm': PETSc.COMM_SELF},
    )()
    try:
        assert matrix.getType() == 'seqdense'
        assert_allclose(matrix.getDenseArray(), expected)
    finally:
        matrix.destroy()
