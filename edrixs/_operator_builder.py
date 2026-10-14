"""Backend-independent kernels for many-body operator matrix entries."""

from __future__ import annotations

import numpy as np

from .fock_basis import FockBasis, FockBinByN


_NUMBA_KERNELS = None


def _operator_terms(emat, umat, norbs, tol_e, tol_u):
    if emat is not None:
        emat = np.asarray(emat)
        if emat.shape != (norbs, norbs):
            raise ValueError(
                "emat has shape {}, expected {}".format(
                    emat.shape, (norbs, norbs)
                )
            )
        iorb, jorb = np.nonzero(np.abs(emat) > tol_e)
        e_terms = np.stack((iorb, jorb), axis=-1).astype(np.int64)
        e_vals = emat[iorb, jorb].astype(np.complex128)
    else:
        e_terms = np.empty((0, 2), dtype=np.int64)
        e_vals = np.empty((0,), dtype=np.complex128)

    if umat is None:
        u_terms = np.empty((0, 4), dtype=np.int64)
        u_vals = np.empty((0,), dtype=np.complex128)
    elif hasattr(umat, 'tocoo') and len(umat.shape) == 2:
        expected = (norbs * norbs, norbs * norbs)
        if umat.shape != expected:
            raise ValueError(
                "sparse umat has shape {}, expected {}".format(umat.shape, expected)
            )
        coo = umat.tocoo()
        keep = np.abs(coo.data) > tol_u
        rows = np.asarray(coo.row[keep], dtype=np.int64)
        cols = np.asarray(coo.col[keep], dtype=np.int64)
        lorb, korb = np.divmod(rows, norbs)
        jorb, iorb = np.divmod(cols, norbs)
        u_terms = np.stack((lorb, korb, jorb, iorb), axis=-1)
        u_vals = np.asarray(coo.data[keep], dtype=np.complex128)
    else:
        umat = np.asarray(umat)
        expected = (norbs, norbs, norbs, norbs)
        if umat.shape != expected:
            raise ValueError(
                "dense umat has shape {}, expected {}".format(umat.shape, expected)
            )
        lorb, korb, jorb, iorb = np.nonzero(np.abs(umat) > tol_u)
        u_terms = np.stack((lorb, korb, jorb, iorb), axis=-1).astype(np.int64)
        u_vals = umat[lorb, korb, jorb, iorb].astype(np.complex128)

    return e_terms, e_vals, u_terms, u_vals


def _sign_count(state, orbital):
    prefix = int(state) & ((1 << orbital) - 1)
    return 1 if prefix.bit_count() % 2 == 0 else -1


def _build_entries_python(lb, rb, e_terms, e_vals, u_terms, u_vals, cstart, cend):
    rows = []
    cols = []
    vals = []

    for column in range(cstart, cend):
        state0 = rb.decode(column)

        for term, value in zip(e_terms, e_vals):
            iorb, jorb = int(term[0]), int(term[1])
            bit_j = 1 << jorb
            if not state0 & bit_j:
                continue
            sign_1 = _sign_count(state0, jorb)
            state = state0 ^ bit_j

            bit_i = 1 << iorb
            if state & bit_i:
                continue
            sign_2 = _sign_count(state, iorb)
            state |= bit_i

            try:
                row = lb.encode(state)
            except KeyError:
                continue
            rows.append(row)
            cols.append(column)
            vals.append(value * sign_1 * sign_2)

        for term, value in zip(u_terms, u_vals):
            lorb, korb, jorb, iorb = map(int, term)
            if iorb == jorb or korb == lorb:
                continue

            bit_i = 1 << iorb
            if not state0 & bit_i:
                continue
            sign_1 = _sign_count(state0, iorb)
            state = state0 ^ bit_i

            bit_j = 1 << jorb
            if not state & bit_j:
                continue
            sign_2 = _sign_count(state, jorb)
            state ^= bit_j

            bit_k = 1 << korb
            if state & bit_k:
                continue
            sign_3 = _sign_count(state, korb)
            state |= bit_k

            bit_l = 1 << lorb
            if state & bit_l:
                continue
            sign_4 = _sign_count(state, lorb)
            state |= bit_l

            try:
                row = lb.encode(state)
            except KeyError:
                continue
            rows.append(row)
            cols.append(column)
            vals.append(value * sign_1 * sign_2 * sign_3 * sign_4)

    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(vals, dtype=np.complex128),
    )


def _get_numba_kernels():
    global _NUMBA_KERNELS
    if _NUMBA_KERNELS is not None:
        return _NUMBA_KERNELS

    try:
        from ._numba_operator_kernels import build_combinadic, build_explicit
    except ImportError as exc:
        raise ImportError(
            "use_numba=True requires numba; install numba or disable JIT construction"
        ) from exc

    _NUMBA_KERNELS = build_combinadic, build_explicit
    return _NUMBA_KERNELS


def _prepare_entries_kernel(
        lb, rb, e_terms, e_vals, u_terms, u_vals, *, use_numba=True):
    """Prepare a reusable entry kernel for backend-selected column ranges."""
    if not use_numba:
        def build_range(cstart, cend):
            return _build_entries_python(
                lb, rb, e_terms, e_vals, u_terms, u_vals, cstart, cend
            )

        return build_range

    if lb.norbs > 64 or rb.norbs > 64:
        raise ValueError("Numba operator construction supports at most 64 orbitals")

    build_combinadic, build_explicit = _get_numba_kernels()

    if isinstance(lb, FockBinByN) and isinstance(rb, FockBinByN):
        rb_meta = rb.jit_args()
        lb_meta = lb.jit_args()

        def build_range(cstart, cend):
            return build_combinadic(
                rb_meta[0], rb_meta[1], rb_meta[2], rb_meta[3], rb_meta[4],
                lb_meta[0], lb_meta[1], lb_meta[2], lb_meta[3], lb_meta[4],
                e_terms, e_vals, u_terms, u_vals, cstart, cend,
            )

        return build_range

    if isinstance(lb, FockBasis) and isinstance(rb, FockBasis):
        try:
            from numba import types
            from numba.typed import Dict
        except ImportError as exc:
            raise ImportError(
                "use_numba=True requires numba; install numba or disable JIT construction"
            ) from exc

        rb_states = np.asarray(rb.basis_int, dtype=np.uint64)
        lb_lookup = Dict.empty(key_type=types.uint64, value_type=types.int64)
        for index, state in enumerate(lb.basis_int):
            lb_lookup[np.uint64(state)] = np.int64(index)

        def build_range(cstart, cend):
            return build_explicit(
                rb_states, lb_lookup,
                e_terms, e_vals, u_terms, u_vals, cstart, cend,
            )

        return build_range

    raise TypeError(
        "Numba construction requires matching explicit or combinadic basis representations"
    )


def prepare_operator_entry_kernel(
        emat, umat, lb, rb=None, *, tol_e=1e-10, tol_u=1e-10,
        use_numba=True):
    """Prepare a reusable kernel that generates entries for requested columns.

    The caller owns the work decomposition and matrix assembly policy. Calling
    the returned function as ``build_range(cstart, cend)`` produces only the
    contributions for that column interval.
    """
    if rb is None:
        rb = lb
    if lb.norbs != rb.norbs:
        raise ValueError("left and right Fock bases must have the same norbs")

    e_terms, e_vals, u_terms, u_vals = _operator_terms(
        emat, umat, rb.norbs, tol_e, tol_u
    )
    return _prepare_entries_kernel(
        lb, rb, e_terms, e_vals, u_terms, u_vals, use_numba=use_numba
    )


def build_operator_entries(
        emat, umat, lb, rb=None, *, tol_e=1e-10, tol_u=1e-10,
        cstart=0, cend=None, use_numba=True):
    """Return entries for one requested column range.

    This compatibility wrapper leaves the work-range choice with the caller;
    backends that need repeated bounded ranges should prepare the reusable
    builder with :func:`prepare_operator_entry_kernel` instead.
    """
    if rb is None:
        rb = lb
    if cend is None:
        cend = len(rb)

    build_range = prepare_operator_entry_kernel(
        emat, umat, lb, rb, tol_e=tol_e, tol_u=tol_u, use_numba=use_numba
    )
    return build_range(cstart, cend)
