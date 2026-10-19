"""Construction, ownership, and phase boundaries for PETSc factories."""

from functools import partial

from operator_factories import operator_factory, operator_factories

import numpy as np
import pytest
from numpy.testing import assert_allclose


def factory(builder):
    result = partial(builder)
    result.backend = 'petsc'
    return result


@pytest.fixture
def petsc_problem():
    PETSc = pytest.importorskip('petsc4py.PETSc')
    from edrixs.petsc_backend import petsc_backend

    initial = np.diag([0., .25, .75, 1.5])
    intermediate = np.array([[2., .125j, .0625, 0], [-.125j, 2.5, .25, .125],
                             [.0625, .25, 3., .125j], [0, .125, -.125j, 3.5]])
    rng = np.random.default_rng(712)
    transitions = [rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)) for _ in range(5)]
    built = []
    peak = [0]

    def build(array, role):
        matrix = PETSc.Mat().createAIJ(array.shape, comm=PETSc.COMM_WORLD)
        built.append((role, matrix))
        peak[0] = max(peak[0], sum(bool(mat.handle) for _, mat in built))
        matrix.setUp()
        start, end = matrix.getOwnershipRange()
        columns = np.arange(array.shape[1], dtype=PETSc.IntType)
        for row in range(start, end):
            matrix.setValues(row, columns, array[row])
        matrix.assemble()
        return matrix

    state = PETSc.Vec().createMPI(4, comm=PETSc.COMM_WORLD)
    state.set(0)
    start, end = state.getOwnershipRange()
    local = state.getArray()
    if start == 0 and end > 0:
        local[0] = 1
    del local
    try:
        yield petsc_backend, initial, intermediate, transitions, state, build, built, peak
    finally:
        state.destroy()
        for _, matrix in built:
            if matrix.handle:
                matrix.destroy()


@pytest.mark.parametrize('lifetime', ['retain', 'recompute'])
@pytest.mark.parametrize('ntrans', [3, 5])
@pytest.mark.parametrize('offset', [0., 2.**40])
@pytest.mark.parametrize('ksp_type', ['gmres', 'shifted_pminres'])
def test_lifetime_policies_match_spectra_and_bound_live_matrices(
        petsc_problem, lifetime, ntrans, offset, ksp_type):
    from edrixs.scipy_backend.scipy_backend import xas_scipy, rixs_scipy

    backend, hi, hn, transitions, state, build, built, peak = petsc_problem
    transitions = transitions[:ntrans]
    arrays = [hi + offset * np.eye(4), hn + offset * np.eye(4), *transitions]
    factories = [factory(partial(build, array, role))
                 for array, role in zip(arrays, ['hi', 'hn', *range(ntrans)])]
    options = {'matrix_lifetime': lifetime, 'nkryl': 4}
    energies = np.array([0.])
    vectors = np.eye(4)[:, :1]
    ominc, eloss = [2.25, 2.75], [-.1, .3, .7]
    channels = [('linear', .2, 'left', 0), ('linear', .2, 'right', 0)]
    expected_x = xas_scipy(energies, vectors, operator_factory(hn), operator_factories(transitions), ominc)
    actual_x = backend.xas_petsc(energies + offset, [state], factories[1],
                                 factories[2:], ominc, backend_kws={'nkryl': 4})
    assert_allclose(actual_x, expected_x, rtol=1e-8, atol=1e-9)
    assert all(not matrix.handle for _, matrix in built)
    built.clear()
    peak[0] = 0
    # skip_gs removes one direction, leaving a three-dimensional final space.
    options.update(ksp_type=ksp_type, linsys_tol=1e-12, nkryl=3)
    expected_r = rixs_scipy(
        energies, vectors, operator_factory(hi), operator_factory(hn),
        operator_factories(transitions), ominc, eloss,
        pol_type=channels, skip_gs=True, return_poles=True,
        backend_kws={'nkryl': 3, 'linsys_tol': 1e-12},
    )
    actual_r = backend.rixs_petsc(
        energies + offset, [state], *factories[:2], factories[2:], ominc, eloss,
        pol_type=channels, skip_gs=True, return_poles=True, backend_kws=options,
    )
    assert_allclose(actual_r[0], expected_r[0], rtol=1e-8, atol=1e-9)
    for actual_channels, expected_channels in zip(actual_r[1], expected_r[1]):
        for actual, expected in zip(actual_channels, expected_channels):
            assert_allclose(actual['norm'], expected['norm'], rtol=1e-8, atol=1e-9)
            assert_allclose(actual['eigval'], np.array(expected['eigval']) + offset)
            for a, b in zip(actual['alpha'], expected['alpha']):
                assert_allclose(a - offset, b, atol=3e-4)
    assert all(not matrix.handle for _, matrix in built)
    if lifetime == 'recompute':
        assert peak[0] == 1
        assert sum(role == 'hn' for role, _ in built) == len(ominc)
        assert sum(role == 'hi' for role, _ in built) == len(ominc) * len(channels)
    else:
        roles = [role for role, _ in built]
        assert len(roles) == len(set(roles))


def test_ed_destroys_factory_matrix_but_preserves_returned_vectors(petsc_problem):
    pytest.importorskip('slepc4py.SLEPc')
    backend, hi, _, _, _, build, built, _ = petsc_problem
    source = factory(partial(build, hi + 1e6 * np.eye(4), 'hi'))
    values, vectors = backend.ed_petsc(
        source, num_evals=2, shift=1e6,
    )
    try:
        assert_allclose(values, [1e6, 1e6 + .25])
        assert all(not matrix.handle for _, matrix in built)
        assert all(vector.norm() > .99 for vector in vectors)
    finally:
        for vector in vectors:
            vector.destroy()


@pytest.mark.parametrize('lifetime', ['retain', 'recompute'])
def test_failure_releases_all_owned_matrices(petsc_problem, lifetime, monkeypatch):
    backend, hi, hn, transitions, state, build, built, _ = petsc_problem
    factories = [factory(partial(build, array, role))
                 for array, role in zip([hi, hn, *transitions[:3]], ['hi', 'hn', 0, 1, 2])]

    def fail(*args, **kwargs):
        raise RuntimeError('Lanczos failed')

    from edrixs.petsc_backend import lanczos
    monkeypatch.setattr(lanczos, 'lanczos_tridiagonal', fail)
    with pytest.raises(RuntimeError, match='Lanczos failed'):
        backend.rixs_petsc([0.], [state], *factories[:2], factories[2:], [2.], [0.],
                           backend_kws={'matrix_lifetime': lifetime, 'ksp_type': 'gmres'})
    assert all(not matrix.handle for _, matrix in built)


@pytest.mark.parametrize('solver', ['xas', 'rixs'])
@pytest.mark.parametrize('bad', [None, 1, 'cached', 'RETAIN'])
def test_invalid_lifetime_rejected_before_construction(solver, bad):
    from edrixs.petsc_backend import petsc_backend

    def forbidden():
        raise AssertionError('Invalid options must not construct matrices')

    source = factory(forbidden)
    expected = TypeError if bad is None or isinstance(bad, int) else ValueError
    args = {
        'xas': [None, None, source, [source] * 3, None],
        'rixs': [None, None, source, source, [source] * 3, None, None],
    }[solver]
    with pytest.raises(expected, match='matrix_lifetime'):
        getattr(petsc_backend, solver + '_petsc')(*args, backend_kws={'matrix_lifetime': bad})


@pytest.mark.parametrize('lifetime', ['retain', 'recompute'])
def test_repeated_factory_arguments_construct_independent_roles(petsc_problem, lifetime):
    from edrixs.scipy_backend.scipy_backend import rixs_scipy

    backend, hi, _, transitions, state, build, built, _ = petsc_problem
    shared = hi + .25 * np.eye(4)
    hamiltonian = factory(partial(build, shared, 'h'))
    transition = factory(partial(build, transitions[0], 't'))
    expected = rixs_scipy(
        [.25], np.eye(4)[:, :1], operator_factory(shared), operator_factory(shared),
        operator_factories([transitions[0]] * 3), [.5, .75], [0., .3],
        backend_kws={'nkryl': 4},
    )
    actual = backend.rixs_petsc(
        [.25], [state], hamiltonian, hamiltonian, [transition] * 3,
        [.5, .75], [0., .3],
        backend_kws={'matrix_lifetime': lifetime, 'ksp_type': 'gmres',
                     'linsys_tol': 1e-12, 'nkryl': 4},
    )
    assert_allclose(actual, expected, atol=1e-9, rtol=1e-8)
    assert all(not matrix.handle for _, matrix in built)
    if lifetime == 'retain':
        assert [role for role, _ in built].count('h') == 2
        assert [role for role, _ in built].count('t') == 2


@pytest.mark.parametrize('lifetime', ['retain', 'recompute'])
def test_zero_rhs_skips_hamiltonians_and_unused_components(petsc_problem, lifetime):
    backend, _, _, _, state, build, built, _ = petsc_problem

    def forbidden():
        raise AssertionError('Unused matrix was constructed')

    unused = factory(forbidden)
    zero = factory(partial(build, np.zeros((4, 4)), 'zero'))
    transitions = [zero, zero, unused]
    # At normal incidence circular polarizations have no z component.
    kwargs = dict(thin=np.pi / 2, backend_kws={'matrix_lifetime': lifetime})
    actual_r = backend.rixs_petsc(
        [0.], [state], unused, unused, transitions, [2.], [0.],
        thout=np.pi / 2, pol_type=[('left', 0, 'left', 0)], **kwargs,
    )
    assert_allclose(actual_r, 0)
    assert all(not matrix.handle for _, matrix in built)


@pytest.mark.parametrize('ntrans', [3, 5])
@pytest.mark.parametrize('offset', [0., 2.**40])
@pytest.mark.parametrize('repeated', [False, True])
@pytest.mark.parametrize('lifetime', [None, 'retain', 'recompute'])
def test_xas_lifetime_spectra(petsc_problem, ntrans, offset, repeated, lifetime):
    from edrixs import xas
    from edrixs.scipy_backend.scipy_backend import xas_scipy

    _, _, hn, transitions, state, build, built, peak = petsc_problem
    transitions = transitions[:ntrans]
    hamiltonian = factory(partial(build, hn + offset * np.eye(4), 'hn'))
    operators = [factory(partial(build, array, component))
                 for component, array in enumerate(transitions)]
    if repeated:
        operators = [operators[0]] * ntrans
        transitions = [transitions[0]] * ntrans
    second = state.duplicate()
    second.set(0)
    start, end = second.getOwnershipRange()
    local = second.getArray()
    if start <= 1 < end:
        local[1 - start] = 1
    del local
    polarizations = [('isotropic', 0), ('linear', .2), ('left', 0), ('right', 0)]
    expected = xas_scipy(
        [0., .25], np.eye(4)[:, :2], operator_factory(hn),
        operator_factories(transitions), [2.25, 2.75],
        pol_type=polarizations, temperature=3000,
    )
    options = {} if lifetime is None else {'matrix_lifetime': lifetime}
    try:
        actual = xas([offset, offset + .25], [state, second], hamiltonian,
                     operators, [2.25, 2.75], pol_type=polarizations,
                     temperature=3000, backend_kws=options)
        assert state.norm() == pytest.approx(1)
        assert second.norm() == pytest.approx(1)
    finally:
        second.destroy()
    assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    roles = [role for role, _ in built]
    if lifetime == 'recompute':
        assert peak[0] == 1
        assert roles.count('hn') == 2 * (ntrans + 3)
        assert len(roles) > ntrans + 1
    else:
        assert len(roles) == ntrans + 1
        assert roles.count('hn') == 1
    assert all(not matrix.handle for _, matrix in built)


@pytest.mark.parametrize('value', ['retain', 'recompute', None])
def test_ed_rejects_matrix_lifetime(value):
    from edrixs.petsc_backend import petsc_backend

    with pytest.raises(TypeError, match='matrix_lifetime'):
        petsc_backend.ed_petsc(None, backend_kws={'matrix_lifetime': value})


@pytest.mark.parametrize('lifetime', ['retain', 'recompute'])
def test_xas_zero_seeds_skip_unused_matrices(petsc_problem, lifetime):
    backend, _, _, _, state, build, built, _ = petsc_problem

    def forbidden():
        raise AssertionError('Unused matrix was constructed')

    zero = factory(partial(build, np.zeros((4, 4)), 'zero'))
    unused = factory(forbidden)
    actual = backend.xas_petsc(
        [0.], [state], unused, [zero, zero, unused], [2.], thin=np.pi / 2,
        pol_type=[('left', 0)], backend_kws={'matrix_lifetime': lifetime},
    )
    assert_allclose(actual, 0)
    assert len(built) == 2
    assert all(not matrix.handle for _, matrix in built)


@pytest.mark.parametrize('lifetime', ['retain', 'recompute'])
@pytest.mark.parametrize('failure', ['factory', 'hamiltonian', 'lanczos', 'mult'])
def test_xas_failure_releases_owned_objects(petsc_problem, lifetime, failure, monkeypatch):
    backend, _, hn, transitions, state, build, built, _ = petsc_problem
    from edrixs.petsc_backend import lanczos

    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')

    vectors = []

    class TrackedMatrix:
        def __init__(self, matrix):
            self.matrix = matrix

        def createVecLeft(self):
            vector = self.matrix.createVecLeft()
            vectors.append(vector)
            return vector

        def mult(self, source, target):
            if failure == 'mult':
                fail()
            self.matrix.mult(source, target)

        def destroy(self):
            self.matrix.destroy()

    def transition(index):
        if failure == 'factory' and index == 1:
            fail()
        return TrackedMatrix(build(transitions[index], index))

    operators = [factory(partial(transition, index)) for index in range(3)]
    hamiltonian = factory(fail if failure == 'hamiltonian' else partial(build, hn, 'hn'))
    if failure == 'lanczos':
        monkeypatch.setattr(lanczos, 'lanczos_tridiagonal', fail)
    with pytest.raises(RuntimeError, match='injected failure'):
        backend.xas_petsc(
            [0.], [state], hamiltonian, operators, [2.], thin=np.pi / 2,
            pol_type=[('left', 0)], backend_kws={'matrix_lifetime': lifetime},
        )
    assert all(not matrix.handle for _, matrix in built)
    assert all(not vector.handle for vector in vectors)
    assert state.norm() == pytest.approx(1)
