__all__ = ['get_s_l_ls_values_i', 'get_s_l_ls_values_n']
import numpy as np

import edrixs


def get_s_l_ls_values_i(l, basis, evec):
    """
    For the ground-state
    Calaculate the  S_x-, S_y-, S_z-, and S^2-Values after diagonalization
    Calaculate the  L_x-, L_y-, L_z-, and L^2-Values after diagonalization
    Calaculate the LS-Values after diagonalization
    Parameters
    ----------

    l: int
        The overall-angular moment l

    evac: 2d complex array
        The impurity vector after demoralization.

    basis: list of array
        Left fock basis :math:`<F_{l}|`.
        AND:
        Right fock basis :math:`|F_{r}>`.
        rb = lb

    Returns
    -------

    spin_prop: list of numpy-arrays
        Expectation-Value of S_x, S_y, S_z, and S^2
    lang_prop: list of numpy-arrays
        Expectation-Value of L_x, L_y, L_z, and L^2
    ls: float-numpy-array
        Expectation-Value of LS
    """
    # Setting up the Spin-Hamiltonian
    h_sx = edrixs.two_fermion(edrixs.get_sx(l=l), basis, basis)
    h_sy = edrixs.two_fermion(edrixs.get_sy(l=l), basis, basis)
    h_sz = edrixs.two_fermion(edrixs.get_sz(l=l), basis, basis)

    # Calculating the S-Expectation-Value depending on the
    # Wavefunction-Coefficient
    sx = edrixs.cb_op2(h_sx, evec, evec)
    sy = edrixs.cb_op2(h_sy, evec, evec)
    sz = edrixs.cb_op2(h_sz, evec, evec)
    # S2-Value has to be calculated as the mutated-product of the single
    # xyz-Hamiltonian consistent with theory
    s2 = edrixs.cb_op2(
        np.matmul(
            h_sx,
            h_sx)
        + np.matmul(
            h_sy,
            h_sy)
        + np.matmul(
            h_sz,
            h_sz),
        evec,
        evec)
    # Summarize everything as a list
    spin_prop = [
        sx.diagonal().real,
        sy.diagonal().real,
        sz.diagonal().real,
        s2.diagonal().real]
    print("Done with Sx-,Sy-,Sz-, and S2-values!")

    # Setting up the Angular-Hamiltonian
    h_lx = edrixs.two_fermion(edrixs.get_lx(l=l, ispin=True), basis, basis)
    h_ly = edrixs.two_fermion(edrixs.get_ly(l=l, ispin=True), basis, basis)
    h_lz = edrixs.two_fermion(edrixs.get_lz(l=l, ispin=True), basis, basis)

    # Calculating the L-Expectation-Value depending on the
    # Wavefunction-Coefficient
    lx = edrixs.cb_op2(h_lx, evec, evec)
    ly = edrixs.cb_op2(h_ly, evec, evec)
    lz = edrixs.cb_op2(h_lz, evec, evec)
    # L2-Value has to be calculated as the mutated-product of the single
    # xyz-Hamiltonian consistent with theory
    l2 = edrixs.cb_op2(
        np.matmul(
            h_lx,
            h_lx)
        + np.matmul(
            h_ly,
            h_ly)
        + np.matmul(
            h_lz,
            h_lz),
        evec,
        evec)
    lang_prop = [
        lx.diagonal().real,
        ly.diagonal().real,
        lz.diagonal().real,
        l2.diagonal().real]
    print("Done with Lx-,Ly-,Lz-, and L2-values!")

    # L2-Value has to be calculated as the mutated-product of the single
    # xyz-Hamiltonian consistent with theory
    ls = edrixs.cb_op2(
        np.matmul(
            h_lx,
            h_sx) +
        np.matmul(
            h_ly,
            h_sy) +
        np.matmul(
            h_lz,
            h_sz),
        evec,
        evec)
    print("Done with LS-values!")

    return spin_prop, lang_prop, ls.diagonal().real


def get_s_l_ls_values_n(l, basis, evec, ref, ndorb, ntot):
    """
    For the 2p to 3d transitions
    Calaculate the  S_x-, S_y-, S_z-, and S^2-Values after diagonalization
    Calaculate the  L_x-, L_y-, L_z-, and L^2-Values after diagonalization
    Calaculate the LS-Values after diagonalization
    Parameters
    ----------

    ndorb: int
        Total number of 3d-electrons

    ntot:
        Total number of electrons

    l: int
        The overall-angular moment l

    evac: 2d complex array
        The impurity vector after demoralization.

    basis: list of array
        Left fock basis :math:`<F_{l}|`.
        AND:
        Right fock basis :math:`|F_{r}>`.
        rb = lb

    Returns
    -------

    spin_prop: list of numpy-arrays
        Expectation-Value of S_x, S_y, S_z, and S^2
    lang_prop: list of numpy-arrays
        Expectation-Value of L_x, L_y, L_z, and L^2
    ls: float-numpy-array
        Expectation-Value of LS
    """

    hh_sx = np.zeros_like(ref)
    hh_sy = np.zeros_like(ref)
    hh_sz = np.zeros_like(ref)

    hh_sx[0:ndorb, 0:ndorb] += edrixs.get_sx(l=l)
    hh_sy[0:ndorb, 0:ndorb] += edrixs.get_sy(l=l)
    hh_sz[0:ndorb, 0:ndorb] += edrixs.get_sz(l=l)
    hh_sx[ndorb:ntot, ndorb:ntot] += edrixs.get_sx(l=l - 1)
    hh_sy[ndorb:ntot, ndorb:ntot] += edrixs.get_sy(l=l - 1)
    hh_sz[ndorb:ntot, ndorb:ntot] += edrixs.get_sz(l=l - 1)
    # Setting up the Spin-Hamiltonian
    h_sx = edrixs.two_fermion(hh_sx, basis, basis)
    h_sy = edrixs.two_fermion(hh_sy, basis, basis)
    h_sz = edrixs.two_fermion(hh_sz, basis, basis)

    # Calculating the S-Expectation-Value depending on the
    # Wavefunction-Coefficient
    sx = edrixs.cb_op2(h_sx, evec, evec)
    sy = edrixs.cb_op2(h_sy, evec, evec)
    sz = edrixs.cb_op2(h_sz, evec, evec)
    # S2-Value has to be calculated as the mutated-product of the single
    # xyz-Hamiltonian consistent with theory
    s2 = edrixs.cb_op2(
        np.matmul(
            h_sx,
            h_sx) +
        np.matmul(
            h_sy,
            h_sy) +
        np.matmul(
            h_sz,
            h_sz),
        evec,
        evec)
    # Summarize everything as a list
    spin_prop = [
        sx.diagonal().real,
        sy.diagonal().real,
        sz.diagonal().real,
        s2.diagonal().real]
    print("Done with Sx-,Sy-,Sz-, and S2-values!")

    hh_lx = np.zeros_like(ref)
    hh_ly = np.zeros_like(ref)
    hh_lz = np.zeros_like(ref)

    hh_lx[0:ndorb, 0:ndorb] += edrixs.get_lx(l=l, ispin=True)
    hh_ly[0:ndorb, 0:ndorb] += edrixs.get_ly(l=l, ispin=True)
    hh_lz[0:ndorb, 0:ndorb] += edrixs.get_lz(l=l, ispin=True)
    hh_lx[ndorb:ntot, ndorb:ntot] += edrixs.get_lx(l=l - 1, ispin=True)
    hh_ly[ndorb:ntot, ndorb:ntot] += edrixs.get_ly(l=l - 1, ispin=True)
    hh_lz[ndorb:ntot, ndorb:ntot] += edrixs.get_lz(l=l - 1, ispin=True)
    # Setting up the Angular-Hamiltonian
    h_lx = edrixs.two_fermion(hh_lx, basis, basis)
    h_ly = edrixs.two_fermion(hh_ly, basis, basis)
    h_lz = edrixs.two_fermion(hh_lz, basis, basis)

    # Calculating the L-Expectation-Value depending on the
    # Wavefunction-Coefficient
    lx = edrixs.cb_op2(h_lx, evec, evec)
    ly = edrixs.cb_op2(h_ly, evec, evec)
    lz = edrixs.cb_op2(h_lz, evec, evec)
    # L2-Value has to be calculated as the mutated-product of the single
    # xyz-Hamiltonian consistent with theory
    l2 = edrixs.cb_op2(
        np.matmul(
            h_lx,
            h_lx) +
        np.matmul(
            h_ly,
            h_ly) +
        np.matmul(
            h_lz,
            h_lz),
        evec,
        evec)
    lang_prop = [
        lx.diagonal().real,
        ly.diagonal().real,
        lz.diagonal().real,
        l2.diagonal().real]
    print("Done with Lx-,Ly-,Lz-, and L2-values!")

    # L2-Value has to be calculated as the mutated-product of the single
    # xyz-Hamiltonian consistent with theory
    ls = edrixs.cb_op2(
        np.matmul(
            h_lx,
            h_sx) +
        np.matmul(
            h_ly,
            h_sy) +
        np.matmul(
            h_lz,
            h_sz),
        evec,
        evec)
    print("Done with LS-values!")

    return spin_prop, lang_prop, ls.diagonal().real


def test_s_d5():
    print(edrixs.get_fock_basis_by_N_LzSz(10, 5, [-2, -2, -1, -1, 0, 0, 1, 1, 2, 2],
                                          [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]))


if __name__ == '__main__':
    test_s_d5()
