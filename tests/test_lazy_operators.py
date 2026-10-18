"""The public solver interface consumes labeled, deferred matrix factories."""

from functools import partial

import numpy as np
import pytest

from edrixs import FockBasisSpec, build_op, ed, get_ops, rixs, xas
from edrixs import solvers
from edrixs.fortran_backend.fortran_backend import FortranDiskOperator


def labeled(builder, backend='scipy'):
    factory = partial(builder)
    factory.backend = backend
    return factory


@pytest.mark.parametrize('backend', ['dense', 'scipy', 'petsc'])
def test_build_op_defers_backend_construction(backend, monkeypatch):
    calls = []
    matrix = object()
    module = getattr(solvers, backend + '_backend')

    def builder(*args, **kwargs):
        calls.append((args, kwargs))
        return matrix

    monkeypatch.setattr(module, 'build_op_' + backend, builder)
    factory = build_op(np.eye(2), None, FockBasisSpec.from_args(2, 1),
                       backend=backend)
    assert isinstance(factory, partial)
    assert factory.backend == backend
    assert calls == []
    assert factory() is matrix
    assert len(calls) == 1


@pytest.mark.parametrize('backend', ['dense', 'scipy'])
def test_factory_builds_fresh_matrices_and_retains_coefficient_references(backend):
    coefficients = np.diag([1., 2.])
    factory = build_op(coefficients, None, FockBasisSpec.from_args(2, 1),
                       backend=backend, use_numba=False)
    first = factory()
    coefficients[0, 0] = 3.
    second = factory()
    assert first is not second
    assert sorted(first.diagonal()) == [1., 2.]
    assert sorted(second.diagonal()) == [2., 3.]


@pytest.mark.parametrize('backend_name', ['dense', 'scipy', 'petsc', 'fortran'])
@pytest.mark.parametrize('solver', ['ed', 'xas', 'rixs'])
def test_public_solver_dispatches_factories_without_building(solver, backend_name, monkeypatch):
    def forbidden():
        raise AssertionError('Public wrapper must not build matrices')

    factory = labeled(forbidden, backend_name)
    result = object()

    def backend(*args, **kwargs):
        if solver == 'ed':
            assert args[0] is factory
        elif solver == 'xas':
            assert args[2] is factory
            assert all(op is factory for op in args[3])
        else:
            assert args[2] is factory and args[3] is factory
            assert all(op is factory for op in args[4])
        return result

    module = getattr(solvers, backend_name + '_backend')
    monkeypatch.setattr(module, solver + '_' + backend_name, backend)
    if solver == 'ed':
        actual = ed(factory)
    elif solver == 'xas':
        actual = xas(None, None, factory, [factory] * 3, None)
    else:
        actual = rixs(None, None, factory, factory, [factory] * 3, None, None)
    assert actual is result


@pytest.mark.parametrize('backend_name', ['dense', 'scipy'])
@pytest.mark.parametrize('solver', ['ed', 'xas', 'rixs'])
def test_backend_eagerly_builds_each_factory_argument(backend_name, solver):
    calls = []
    factory = partial(lambda: calls.append(1) or np.diag([1., 2.]))
    module = getattr(solvers, backend_name + '_backend')
    run = getattr(module, solver + '_' + backend_name)
    for _ in range(2):
        if solver == 'ed':
            run(factory, num_evals=2)
        elif solver == 'xas':
            run([0.], np.eye(2)[:, :1], factory, [factory] * 3, [1.])
        else:
            run([0.], np.eye(2)[:, :1], factory, factory, [factory] * 3, [1.], [0.])
    assert calls == [1] * (2 * {'ed': 1, 'xas': 4, 'rixs': 5}[solver])


@pytest.mark.parametrize('backend_name', ['dense', 'scipy'])
def test_backend_builds_all_inputs_before_calculation(backend_name, monkeypatch):
    calls = []
    factories = [labeled(lambda i=i: calls.append(i) or np.eye(2), backend_name)
                 for i in range(5)]
    module = getattr(solvers, backend_name + '_backend')
    if backend_name == 'dense':
        def stop(*args, **kwargs):
            assert calls == list(range(5))
            raise RuntimeError('calculation started')

        monkeypatch.setattr(module.scipy.linalg, 'eigh', stop)
    else:
        def stop(*args, **kwargs):
            assert calls == list(range(5))
            raise RuntimeError('calculation started')

        monkeypatch.setattr(module, '_shift_hamiltonian', stop)
    with pytest.raises(RuntimeError, match='calculation started'):
        getattr(module, 'rixs_' + backend_name)(
            [0.], np.eye(2)[:, :1], *factories[:2], factories[2:], [1.], [0.]
        )


@pytest.mark.parametrize('solver', ['ed', 'xas', 'rixs'])
def test_fortran_handles_pass_through_without_calling(solver, monkeypatch):
    handle = FortranDiskOperator()
    assert handle.backend == 'fortran'
    assert handle() is handle
    result = object()

    def backend(*args, **kwargs):
        index = 0 if solver == 'ed' else 2
        assert args[index] is handle
        return result
    monkeypatch.setattr(solvers.fortran_backend, solver + '_fortran', backend)
    if solver == 'ed':
        actual = ed(handle)
    elif solver == 'xas':
        actual = xas(None, None, handle, [handle] * 3, None)
    else:
        actual = rixs(None, None, handle, handle, [handle] * 3, None, None)
    assert actual is result


def test_solver_backend_keyword_removed():
    factory = labeled(lambda: np.eye(2))
    for solver, args in [(ed, [factory]),
                         (xas, [None, None, factory, [factory] * 3, None]),
                         (rixs, [None, None, factory, factory, [factory] * 3, None, None])]:
        with pytest.raises(TypeError, match="unexpected keyword argument 'backend'"):
            solver(*args, backend='scipy')


def test_backend_options_validation_occurs_when_building():
    factory = build_op(np.eye(2), None, FockBasisSpec.from_args(2, 1),
                       backend_kws={'unknown_option': 1})
    with pytest.raises(TypeError, match='Unknown backend_kws'):
        factory()


def test_ed_rejects_nonmapping_solver_options():
    factory = labeled(lambda: np.eye(2))
    with pytest.raises(TypeError, match='mapping'):
        ed(factory, backend_kws=[('tol', 1e-8)])


def test_get_ops_returns_labeled_factories(small_1v1c_problem):
    hi, hn, transitions = get_ops(*small_1v1c_problem[:7], use_numba=False)
    assert all(isinstance(op, partial) and op.backend == 'scipy'
               for op in [hi, hn, *transitions])
    values, vectors = ed(hi, num_evals=1)
    assert len(values) == 1
    assert vectors.shape[1] == 1


@pytest.mark.parametrize('solver', ['xas', 'rixs'])
def test_dispatch_reads_only_primary_hamiltonian_label(solver, monkeypatch):
    primary = labeled(lambda: None, 'scipy')
    other = object()
    result = object()

    def backend(*args, **kwargs):
        assert args[2] is primary
        if solver == 'rixs':
            assert args[3] is other
        else:
            assert args[3] == [other] * 3
        return result

    monkeypatch.setattr(solvers.scipy_backend, solver + '_scipy', backend)
    if solver == 'xas':
        actual = xas(None, None, primary, [other] * 3, None)
    else:
        actual = rixs(None, None, primary, other, [other] * 3, None, None)
    assert actual is result
