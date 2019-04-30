__all__ = ['one_fermion_annihilation', 'two_fermion', 'four_fermion', 'density_matrix']

import numpy as np
from collections import defaultdict


def one_fermion_annihilation(iorb, lb, rb):
    """
    Build matrix form of a fermionic annihilation operator in the given Fock basis.

    .. math::

        <F_{l}|\\hat{f}_{i}|F_{r}>

    Parameters
    ----------
    iorb: int
        Which orbital.
    lb: list or array
        Left fock basis :math:`<F_{l}|`.
    rb: list of array
        Right fock basis :math:`|F_{r}>`.

    Returns
    -------
    hmat: 2d complex array
        The matrix form of :math:`\\hat{f}_{i}`.
    """

    lb, rb = np.array(lb), np.array(rb)
    nr, nl, norbs = len(rb), len(lb), len(rb[0])
    indx = defaultdict(lambda: -1)
    for i, j in enumerate(lb):
        indx[tuple(j)] = i

    hmat = np.zeros((nl, nr), dtype=np.complex128)
    tmp_basis = np.zeros(norbs)
    for icfg in range(nr):
        tmp_basis[:] = rb[icfg]
        if tmp_basis[iorb] == 0:
            continue
        else:
            sign = (-1)**np.count_nonzero(tmp_basis[0:iorb])
            tmp_basis[iorb] = 0
        jcfg = indx[tuple(tmp_basis)]
        if jcfg != -1:
            hmat[jcfg, icfg] += sign
    return hmat


def two_fermion(emat, lb, rb, tol=1E-10):
    """
    Build matrix form of a two-fermionic operator in the given Fock basis,

    .. math::

        <F_{l}|\\sum_{ij}E_{ij}\\hat{f}_{i}^{\\dagger}\\hat{f}_{j}|F_{r}>

    Parameters
    ----------
    emat: 2d complex array
        The impurity matrix.
    lb: list or array
        Left fock basis :math:`<F_{l}|`.
    rb: list of array
        Right fock basis :math:`|F_{r}>`.
    tol: float (default: 1E-10)
        Only consider the elements of emat that are larger than tol.

    Returns
    -------
    hmat: 2d complex array
        The matrix form of the two-fermionic operator.
    """

    lb, rb = np.array(lb), np.array(rb)
    nr, nl, norbs = len(rb), len(lb), len(rb[0])
    indx = defaultdict(lambda: -1)
    for i, j in enumerate(lb):
        indx[tuple(j)] = i

    a1, a2 = np.nonzero(abs(emat) > tol)
    nonzero = np.stack((a1, a2), axis=-1)

    hmat = np.zeros((nl, nr), dtype=np.complex128)
    tmp_basis = np.zeros(norbs)
    for iorb, jorb in nonzero:
        for icfg in range(nr):
            tmp_basis[:] = rb[icfg]
            if tmp_basis[jorb] == 0:
                continue
            else:
                s1 = (-1)**np.count_nonzero(tmp_basis[0:jorb])
                tmp_basis[jorb] = 0
            if tmp_basis[iorb] == 1:
                continue
            else:
                s2 = (-1)**np.count_nonzero(tmp_basis[0:iorb])
                tmp_basis[iorb] = 1
            jcfg = indx[tuple(tmp_basis)]
            if jcfg != -1:
                hmat[jcfg, icfg] += emat[iorb, jorb] * s1 * s2
    return hmat


def four_fermion(umat, basis, tol=1E-10):
    """
    Build matrix form of a four-fermionic operator in the given Fock basis,

    .. math::

        <F|\\sum_{ij}U_{ijkl}\\hat{f}_{i}^{\\dagger}\\hat{f}_{j}^{\\dagger}\\hat{f}_{k}\\hat{f}_{l}|F>

    Parameters
    ----------
    umat: 4d complex array
        The 4 index Coulomb interaction tensor.
    basis: list or array
        Fock basis :math:`|F>`.
    tol: float (default: 1E-10)
        Only consider the elements of umat that are larger than tol.

    Returns
    -------
    hmat: 2d complex array
        The matrix form of the four-fermionic operator.
    """

    basis = np.array(basis)
    ncfgs, norbs = len(basis), len(basis[0])
    indx = defaultdict(lambda: -1)
    for i, j in enumerate(basis):
        indx[tuple(j)] = i

    a1, a2, a3, a4 = np.nonzero(abs(umat) > tol)
    nonzero = np.stack((a1, a2, a3, a4), axis=-1)

    hmat = np.zeros((ncfgs, ncfgs), dtype=np.complex128)
    tmp_basis = np.zeros(norbs)
    for lorb, korb, jorb, iorb in nonzero:
        if iorb == jorb or korb == lorb:
            continue
        for icfg in range(ncfgs):
            tmp_basis[:] = basis[icfg]
            if tmp_basis[iorb] == 0:
                continue
            else:
                s1 = (-1)**np.count_nonzero(tmp_basis[0:iorb])
                tmp_basis[iorb] = 0
            if tmp_basis[jorb] == 0:
                continue
            else:
                s2 = (-1)**np.count_nonzero(tmp_basis[0:jorb])
                tmp_basis[jorb] = 0
            if tmp_basis[korb] == 1:
                continue
            else:
                s3 = (-1)**np.count_nonzero(tmp_basis[0:korb])
                tmp_basis[korb] = 1
            if tmp_basis[lorb] == 1:
                continue
            else:
                s4 = (-1)**np.count_nonzero(tmp_basis[0:lorb])
                tmp_basis[lorb] = 1
            jcfg = indx[tuple(tmp_basis)]
            if jcfg != -1:
                hmat[jcfg, icfg] += umat[lorb, korb, jorb, iorb] * s1 * s2 * s3 * s4
    return hmat


def density_matrix(iorb, jorb, lb, rb):
    """
    Calculate the matrix form of density operators :math:`\\hat{f}_{i}^{\\dagger}\\hat{f}_{j}`
    in the given Fock basis,

    .. math::

        <F_{l}|\\hat{f}_{i}^{\\dagger}\\hat{f}_{j}|F_{r}>

    Parameters
    ----------
    iorb: int
        Orbital index.
    jorb: int
        Orbital index.
    lb: list or array
        Left fock basis :math:`<F_{l}|`.
    rb: list or array
        Right fock basis :math:`|F_{r}>`.

    Returns
    -------
    hmat: 2d complex array
        The calculated matrix form of the density operator.
    """

    lb, rb = np.array(lb), np.array(rb)
    nr, nl, norbs = len(rb), len(lb), len(rb[0])
    indx = defaultdict(lambda: -1)
    for i, j in enumerate(lb):
        indx[tuple(j)] = i

    hmat = np.zeros((nl, nr), dtype=np.complex128)
    tmp_basis = np.zeros(norbs)
    for icfg in range(nr):
        tmp_basis[:] = rb[icfg]
        if tmp_basis[jorb] == 0:
            continue
        else:
            s1 = (-1)**np.count_nonzero(tmp_basis[0:jorb])
            tmp_basis[jorb] = 0
        if tmp_basis[iorb] == 1:
            continue
        else:
            s2 = (-1)**np.count_nonzero(tmp_basis[0:iorb])
            tmp_basis[iorb] = 1
        jcfg = indx[tuple(tmp_basis)]
        if jcfg != -1:
            hmat[jcfg, icfg] += s1 * s2
    return hmat
