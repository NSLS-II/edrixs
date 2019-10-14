__all__ = ['get_s_l_ls_values_i', 'get_s_l_ls_values_n']
import numpy as np

import edrixs


def get_s_l_ls_values_i(l_ang, basis, evec):
    """
    For the ground-state
    Calaculate the  S_x-, S_y-, S_z-, and S^2-Values after diagonalization
    Calaculate the  L_x-, L_y-, L_z-, and L^2-Values after diagonalization
    Calaculate the LS-Values after diagonalization

    Parameters
    ----------

    l_ang: int
        The overall-angular moment l
    basis: list of array
        Left fock basis :math:`<F_{l}|`.
        AND:
        Right fock basis :math:`|F_{r}>`.
        rb = lb
    evec: 2d complex array
        The impurity vector after demoralization.

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
    h_sx = edrixs.two_fermion(edrixs.get_sx(l_ang), basis, basis)
    h_sy = edrixs.two_fermion(edrixs.get_sy(l_ang), basis, basis)
    h_sz = edrixs.two_fermion(edrixs.get_sz(l_ang), basis, basis)

    # Calculating the S-Expectation-Value depending on the
    # Wavefunction-Coefficient
    sx = edrixs.cb_op2(h_sx, evec, evec)
    sy = edrixs.cb_op2(h_sy, evec, evec)
    sz = edrixs.cb_op2(h_sz, evec, evec)
    # S2-Value has to be calculated as the mutated-product of the single
    # xyz-Hamiltonian consistent with theory
    s2 = np.dot(sx, sx) + np.dot(sy, sy) + np.dot(sz, sz)
    # Summarize everything as a list
    spin_prop = [
        sx.diagonal().real,
        sy.diagonal().real,
        sz.diagonal().real,
        s2.diagonal().real]
    print("Done with Sx-,Sy-,Sz-, and S2-values!")

    # Setting up the Angular-Hamiltonian
    h_lx = edrixs.two_fermion(edrixs.get_lx(l_ang, ispin=True), basis, basis)
    h_ly = edrixs.two_fermion(edrixs.get_ly(l_ang, ispin=True), basis, basis)
    h_lz = edrixs.two_fermion(edrixs.get_lz(l_ang, ispin=True), basis, basis)

    # Calculating the L-Expectation-Value depending on the
    # Wavefunction-Coefficient
    lx = edrixs.cb_op2(h_lx, evec, evec)
    ly = edrixs.cb_op2(h_ly, evec, evec)
    lz = edrixs.cb_op2(h_lz, evec, evec)
    # L2-Value has to be calculated as the mutated-product of the single
    # xyz-Hamiltonian consistent with theory
    l2 = np.dot(lx, lx) + np.dot(ly, ly) + np.dot(lz, lz)
    lang_prop = [
        lx.diagonal().real,
        ly.diagonal().real,
        lz.diagonal().real,
        l2.diagonal().real]
    print("Done with Lx-,Ly-,Lz-, and L2-values!")

    # L2-Value has to be calculated as the mutated-product of the single
    # xyz-Hamiltonian consistent with theory
    ls = np.dot(lx, sx) + np.dot(ly, sy) + np.dot(lz, sz)
    print("Done with LS-values!")

    return spin_prop, lang_prop, ls.diagonal().real


def get_s_l_ls_values_n(l_ang, basis, evec, ref, ndorb, ntot):
    """
    For the core to excited state transitions
    Calaculate the  S_x-, S_y-, S_z-, and S^2-Values after diagonalization
    Calaculate the  L_x-, L_y-, L_z-, and L^2-Values after diagonalization
    Calaculate the LS-Values after diagonalization

    Parameters
    ----------

    l_ang: int
        The overall-angular moment l
    basis: list of array
        Left fock basis :math:`<F_{l}|`.
        AND:
        Right fock basis :math:`|F_{r}>`.
        rb = lb
    evec: 2d complex array
        The impurity vector after demoralization.
    ref: 2d complex array
        Excited state Model-Hamilonian
    ndorb: int
        Total number of acceptor orbitals
    ntot: int
        Total number of orbitals


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

    hh_sx[0:ndorb, 0:ndorb] += edrixs.get_sx(l_ang)
    hh_sy[0:ndorb, 0:ndorb] += edrixs.get_sy(l_ang)
    hh_sz[0:ndorb, 0:ndorb] += edrixs.get_sz(l_ang)
    hh_sx[ndorb:ntot, ndorb:ntot] += edrixs.get_sx(l_ang - 1)
    hh_sy[ndorb:ntot, ndorb:ntot] += edrixs.get_sy(l_ang - 1)
    hh_sz[ndorb:ntot, ndorb:ntot] += edrixs.get_sz(l_ang - 1)
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
    s2 = np.dot(sx, sx) + np.dot(sy, sy) + np.dot(sz, sz)
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

    hh_lx[0:ndorb, 0:ndorb] += edrixs.get_lx(l_ang, ispin=True)
    hh_ly[0:ndorb, 0:ndorb] += edrixs.get_ly(l_ang, ispin=True)
    hh_lz[0:ndorb, 0:ndorb] += edrixs.get_lz(l_ang, ispin=True)
    hh_lx[ndorb:ntot, ndorb:ntot] += edrixs.get_lx(l_ang - 1, ispin=True)
    hh_ly[ndorb:ntot, ndorb:ntot] += edrixs.get_ly(l_ang - 1, ispin=True)
    hh_lz[ndorb:ntot, ndorb:ntot] += edrixs.get_lz(l_ang - 1, ispin=True)
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
    l2 = np.dot(lx, lx) + np.dot(ly, ly) + np.dot(lz, lz)
    lang_prop = [
        lx.diagonal().real,
        ly.diagonal().real,
        lz.diagonal().real,
        l2.diagonal().real]
    print("Done with Lx-,Ly-,Lz-, and L2-values!")

    # L2-Value has to be calculated as the mutated-product of the single
    # xyz-Hamiltonian consistent with theory
    ls = np.dot(lx, sx) + np.dot(ly, sy) + np.dot(lz, sz)
    print("Done with LS-values!")

    return spin_prop, lang_prop, ls.diagonal().real
