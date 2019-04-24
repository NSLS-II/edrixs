__all__ = ['ed_1v1c', 'xas_1v1c', 'rixs_1v1c', 'ed_2v1c', 'xas_2v1c']

import numpy as np
import scipy
from .utils import info_atomic_shell, slater_integrals_name, boltz_dist
from .coulomb_utensor import get_umat_slater, get_umat_slater_3shells
from .iostream import write_tensor, write_emat, write_umat, write_config
from .iostream import read_poles_from_file, rename_pole_file
from .plot_spectrum import get_spectra_from_poles
from .soc import atom_hsoc
from .angular_momentum import get_sx, get_sy, get_sz, get_lx, get_ly, get_lz
from .angular_momentum import rmat_to_euler, get_wigner_dmat
from .manybody_operator import two_fermion, four_fermion
from .fock_basis import get_fock_bin_by_N, write_fock_dec_by_N
from .basis_transform import cb_op2, tmat_r2c
from .photon_transition import get_trans_oper, quadrupole_polvec
from .photon_transition import dipole_polvec_xas, dipole_polvec_rixs, unit_wavevector
from .rixs_utils import scattering_mat
from .fedrixs import ed_fsolver, xas_fsolver


def ed_1v1c(v_name='d', c_name='p', v_level=0.0, c_level=0.0,
            v_soc=(0.0, 0.0), c_soc=0.0,
            v_noccu=1, slater=([0], [0]),
            ext_B=np.zeros(3), zeeman_on_which='spin',
            cf_mat=None, other_mat=None,
            local_axis=np.eye(3), verbose=0):
    """
    Perform ED for two atomic shell case, one Valence-shell plus one Core-shell.
    For example, for Ni-L3 edge RIXS, they are 3d valence shell and 2p core shell.

    Parameters
    ----------
    v_name: string
        Valence shell type, can be 's', 'p', 't2g', 'd', 'f'.
    c_name: string
        Core shell type, can be 's', 'p', 'p12', 'p32', 'd', 'd32', 'd52', 'f', 'f52', 'f72'.
    v_level: float number
        Energy level of valence shell.
    c_level: float number
        Energy level of core shell.
    v_soc: tuple of two float numbers
        Spin-orbit coupling strength of valence electrons,
        v_soc[0] for the initial Hamiltonian, and
        v_soc[1] for the intermediate Hamiltonian.
    c_soc: float number
        Spin-orbit coupling strength of core electrons.
    v_noccu: int number
        Number of electrons in valence shell.
    slater: tuple of two lists
        Slater integrals for initinal (slater[0]) and intermediate (slater[1]) Hamiltonian.
        The order of the elements in the lists like this: [FX_vv, FX_vc, GX_vc, FX_cc], where
        X are integers with increasing order, it can be X=0, 2, 4, 6 or X=1, 3, 5.
    ext_B: list of three float numbers
        Vector of external magnetic field, with respect to global :math:`xyz`-axis.
    zeeman_on_which: string
        Apply Zeeman exchange field on 'spin', 'orbital' or 'both'.
    cf_mat: 2d complex array
        Crystal field splitting Hamiltonian of valence electrons in the complex harmonics
        basis with spin degree of freedom. The orbital order is:
        :math:`|-l,\\uparrow>, |-l,\\downarrow>, |-l+1,\\uparrow>, |-l+1,\\downarrow>, ...,
        |l,\\uparrow>, |l,\\downarrow>`.
    other_mat: 2d complex array
        OtherHamiltonian of valence electrons in the complex harmonics
        basis with spin degree of freedom. The orbital order is:
        :math:`|-l,\\uparrow>, |-l,\\downarrow>, |-l+1,\\uparrow>, |-l+1,\\downarrow>, ...,
        |l,\\uparrow>, |l,\\downarrow>`.
    local_axis: 3*3 float array
        The local axis with respect to which the orbitals are defined.

        x: local_axis[:,0],

        y: local_axis[:,1],

        z: local_axis[:,2].
    verbose: int
        Level of dumping files.

    Returns
    -------
    eval_i:  1d float array
        The eigenvalues of initial Hamiltonian.
    eval_n: 1d float array
        The eigenvalues of intermediate Hamiltonian.
    trans_op: 3d complex array
        The matrices of transition operators in the eigenvector basis,
        and they are defined with respect to the global :math:`xyz`-axis.
    """
    print("edrixs >>> Running ED ...")
    v_name = v_name.strip()
    c_name = c_name.strip()
    info_shell = info_atomic_shell()
    # Quantum numbers of angular momentum
    v_orbl = info_shell[v_name][0]

    # number of orbitals with spin
    v_norb = info_shell[v_name][1]
    c_norb = info_shell[c_name][1]
    # total number of orbitals
    ntot = v_norb + c_norb

    emat_i = np.zeros((ntot, ntot), dtype=np.complex)
    emat_n = np.zeros((ntot, ntot), dtype=np.complex)

    # Coulomb interaction
    slater_name = slater_integrals_name((v_name, c_name), ('v', 'c'))
    num_slat = len(slater_name)
    slater_i = np.zeros(num_slat, dtype=np.float)
    slater_n = np.zeros(num_slat, dtype=np.float)
    if num_slat > len(slater[0]):
        slater_i[0:len(slater[0])] = slater[0]
    else:
        slater_i[:] = slater[0][0:num_slat]
    if num_slat > len(slater[1]):
        slater_n[0:len(slater[1])] = slater[1]
    else:
        slater_n[:] = slater[1][0:num_slat]

    # print summary of slater integrals
    print("    Summary of Slater integrals:")
    print("    ------------------------------")
    print("    Terms,  Initial,  Intermediate")
    for i in range(num_slat):
        print("    ", slater_name[i], ":  {:20.10f}{:20.10f}".format(slater_i[i], slater_n[i]))

    case = v_name + c_name
    umat_i = get_umat_slater(case, *slater_i)
    umat_n = get_umat_slater(case, *slater_n)

    if verbose == 1:
        write_umat(umat_i, 'coulomb_i.in')
        write_umat(umat_n, 'coulomb_n.in')

    # SOC
    emat_i[0:v_norb, 0:v_norb] += atom_hsoc(v_name, v_soc[0])
    emat_n[0:v_norb, 0:v_norb] += atom_hsoc(v_name, v_soc[1])
    # If only sub-core-shell, p12, p32, d32, d52, f52, f72,
    # do not need add SOC
    if c_name in ['p', 'd', 'f']:
        emat_n[v_norb:ntot, v_norb:ntot] += atom_hsoc(c_name, c_soc)

    # crystal field
    if cf_mat is not None:
        emat_i[0:v_norb, 0:v_norb] += cf_mat
        emat_n[0:v_norb, 0:v_norb] += cf_mat

    # other hopping matrix
    if other_mat is not None:
        emat_i[0:v_norb, 0:v_norb] += other_mat
        emat_n[0:v_norb, 0:v_norb] += other_mat

    # energy of shell
    emat_i[0:v_norb, 0:v_norb] += np.eye(v_norb) * v_level
    emat_i[v_norb:ntot, v_norb:ntot] += np.eye(c_norb) * c_level
    emat_n[0:v_norb, 0:v_norb] += np.eye(v_norb) * v_level
    emat_n[v_norb:ntot, v_norb:ntot] += np.eye(c_norb) * c_level

    # external magnetic field
    if v_name == 't2g':
        lx, ly, lz = get_lx(1, True), get_ly(1, True), get_lz(1, True)
        sx, sy, sz = get_sx(1), get_sy(1), get_sz(1)
        lx, ly, lz = -lx, -ly, -lz
    else:
        lx, ly, lz = get_lx(v_orbl, True), get_ly(v_orbl, True), get_lz(v_orbl, True)
        sx, sy, sz = get_sx(v_orbl), get_sy(v_orbl), get_sz(v_orbl)

    if zeeman_on_which.strip() == 'spin':
        zeeman = ext_B[0] * (2 * sx) + ext_B[1] * (2 * sy) + ext_B[2] * (2 * sz)
    elif zeeman_on_which.strip() == 'orbital':
        zeeman = ext_B[0] * lx + ext_B[1] * ly + ext_B[2] * lz
    elif zeeman_on_which.strip() == 'both':
        zeeman = ext_B[0] * (lx + 2 * sx) + ext_B[1] * (ly + 2 * sy) + ext_B[2] * (lz + 2 * sz)
    else:
        raise Exception("Unknown value of zeeman_on_which")

    emat_i[0:v_norb, 0:v_norb] += zeeman
    emat_n[0:v_norb, 0:v_norb] += zeeman

    if verbose == 1:
        write_emat(emat_i, 'hopping_i.in')
        write_emat(emat_n, 'hopping_n.in')

    basis_i = get_fock_bin_by_N(v_norb, v_noccu, c_norb, c_norb)
    basis_n = get_fock_bin_by_N(v_norb, v_noccu+1, c_norb, c_norb - 1)
    ncfg_i, ncfg_n = len(basis_i), len(basis_n)
    print("    Dimension of the initial Hamiltonian: ", ncfg_i)
    print("    Dimension of the intermediate Hamiltonian: ", ncfg_n)

    # Build many-body Hamiltonian in Fock basis
    print("edrixs >>> Building Many-body Hamiltonians ...")
    hmat_i = np.zeros((ncfg_i, ncfg_i), dtype=np.complex)
    hmat_n = np.zeros((ncfg_n, ncfg_n), dtype=np.complex)
    hmat_i[:, :] += two_fermion(emat_i, basis_i, basis_i)
    hmat_i[:, :] += four_fermion(umat_i, basis_i)
    hmat_n[:, :] += two_fermion(emat_n, basis_n, basis_n)
    hmat_n[:, :] += four_fermion(umat_n, basis_n)
    print("edrixs >>> Done !")

    # Do exact-diagonalization to get eigenvalues and eigenvectors
    print("edrixs >>> Exact Diagonalization of Hamiltonians ...")
    eval_i, evec_i = scipy.linalg.eigh(hmat_i)
    eval_n, evec_n = scipy.linalg.eigh(hmat_n)
    print("edrixs >>> Done !")

    if verbose == 1:
        write_tensor(eval_i, 'eval_i.dat')
        write_tensor(eval_n, 'eval_n.dat')

    # Build dipolar transition operators in local-xyz axis
    tmp = get_trans_oper(case)
    npol, n, m = tmp.shape
    tmp_g = np.zeros((npol, n, m), dtype=np.complex)
    # Transform the transition operators to global-xyz axis
    # dipolar transition
    if npol == 3:
        for i in range(3):
            for j in range(3):
                tmp_g[i] += local_axis[i, j] * tmp[j]

    # quadrupolar transition
    elif npol == 5:
        alpha, beta, gamma = rmat_to_euler(local_axis)
        wignerD = get_wigner_dmat(4, alpha, beta, gamma)
        rotmat = np.dot(np.dot(tmat_r2c('d'), wignerD), np.conj(np.transpose(tmat_r2c('d'))))
        for i in range(5):
            for j in range(5):
                tmp_g[i] += rotmat[i, j] * tmp[j]
    else:
        raise Exception("Have NOT implemented this case: ", npol)

    tmp2 = np.zeros((npol, ntot, ntot), dtype=np.complex)
    trans_op = np.zeros((npol, ncfg_n, ncfg_i), dtype=np.complex)
    for i in range(npol):
        tmp2[i, 0:v_norb, v_norb:ntot] = tmp_g[i]
        trans_op[i] = two_fermion(tmp2[i], basis_n, basis_i)
        trans_op[i] = cb_op2(trans_op[i], evec_n, evec_i)

    print("edrixs >>> ED Done !")
    return eval_i, eval_n, trans_op


def xas_1v1c(eval_i, eval_n, trans_op, ominc_mesh,
             gamma_c=0.1,
             thin=1.0,
             phi=0.0,
             poltype=[('isotropic', 0.0)],
             gs_list=[0],
             temperature=1.0,
             scattering_plane_axis=np.eye(3)
             ):

    """
    Calculate XAS for one valence shell plus one core shell.

    Parameters
    ----------
    eval_i: 1d float array
        The eigenvalues of the initial Hamiltonian.
    eval_n: 1d float array
        The eigenvalues of the intermediate Hamiltonian.
    trans_op: 3d complex array
        The transition operators in the eigenstates basis.
    ominc_mesh: 1d float array
        The mesh of the incident energy of photon.
    gamma_c: a float number or a 1d float array with same length as ominc_mesh.
        The core-hole life-time broadening factor. It can be a constant value
        or incident energy dependent.
    thin: float number
        The incident angle of photon.
    phi: float number
        Azimuthal angle in radian, defined with respect to the
        :math:`x`-axis: scattering_plane_axis[:,0].
    poltype: list of tuples
        Type of polarization, options are
        ('linear', alpha), where alpha is the angle between the polarization vector and
        the scattering plane. ('left', 0), ('right', 0), ('isotropic', 0).
    gs_list: 1d int list
        The indices of initial states which will be used as initial states in XAS calculations.
    temperature: float number
        Temperature (in K) for boltzmann distribution.
    scattering_plane_axis: 3*3 float array
        The local axis defining the scattering plane. The scattering plane is defined in
        the local :math:`zx`-plane.
        local :math:`x`-axis: scattering_plane_axis[:,0]
        local :math:`y`-axis: scattering_plane_axis[:,1]
        local :math:`z`-axis: scattering_plane_axis[:,2]

    Returns
    -------
    xas: 2d float array
        The calculated XAS spectra.
    """

    print("edrixs >>> Running XAS ...")
    n_om = len(ominc_mesh)
    npol, ncfg_n = trans_op.shape[0], trans_op.shape[1]
    xas = np.zeros((n_om, len(poltype)), dtype=np.float)
    gamma_core = np.zeros(n_om, dtype=np.float)
    prob = boltz_dist([eval_i[i] for i in gs_list], temperature)
    if np.isscalar(gamma_c):
        gamma_core[:] = np.ones(n_om) * gamma_c
    else:
        gamma_core[:] = gamma_c

    kvec = unit_wavevector(thin, phi, scattering_plane_axis, 'in')
    for i, om in enumerate(ominc_mesh):
        for it, (pt, alpha) in enumerate(poltype):
            polvec = np.zeros(npol, dtype=np.complex)
            if pt.strip() == 'isotropic':
                pol = np.ones(3)/np.sqrt(3.0)
            elif pt.strip() == 'left' or pt.strip() == 'right' or pt.strip() == 'linear':
                pol = dipole_polvec_xas(thin, phi, alpha, scattering_plane_axis, pt)
            else:
                raise Exception("Unknown polarization type: ", pt)

            if npol == 3:  # Dipolar transition
                polvec[:] = pol
            if npol == 5:  # Quadrupolar transition
                polvec[:] = quadrupole_polvec(pol, kvec)
            # loop over all the initial states
            for j in gs_list:
                F_mag = np.zeros(ncfg_n, dtype=np.complex)
                for k in range(npol):
                    F_mag += trans_op[k, :, j] * polvec[k]
                xas[i, it] += (prob[j] * np.sum(np.abs(F_mag)**2 * gamma_core[i] / np.pi /
                               ((om - (eval_n[:] - eval_i[j]))**2 + gamma_core[i]**2)))

    print("edrixs >>> XAS Done !")
    return xas


def rixs_1v1c(eval_i, eval_n, trans_op, ominc_mesh, eloss_mesh,
              gamma_c=0.1,
              gamma_f=0.01,
              thin=1.0,
              thout=1.0,
              phi=0.0,
              poltype=[('linear', 0.0, 'linear', 0.0)],
              gs_list=[0],
              temperature=1.0,
              scattering_plane_axis=np.eye(3)
              ):
    """
    Calculate RIXS for one valence shell plus one core shell.

    Parameters
    ----------
    eval_i: 1d float array
        The eigenvalues of the initial Hamiltonian.
    eval_n: 1d float array
        The eigenvalues of the intermediate Hamiltonian.
    trans_op: 3d complex array
        The transition operators in the eigenstates basis.
    ominc_mesh: 1d float array
        The mesh of the incident energy of photon.
    eloss_mesh: 1d float array
        The mesh of energy loss.
    gamma_c: a float number or a 1d float array with same length as ominc_mesh.
        The core-hole life-time broadening factor. It can be a constant value
        or incident energy dependent.
    gamma_f: a float number or a 1d float array with same length as eloss_mesh.
        The final states life-time broadening factor. It can be a constant value
        or energy loss dependent.
    thin: float number
        The incident angle of photon.
    thout: float number
        The scattered angle of photon.
    phi: float number
        Azimuthal angle in radian, defined with respect to the
        :math:`x`-axis: scattering_plane_axis[:,0].
    poltype: list of 4-elements-tuples
        Type of polarizations.
    gs_list: 1d int list
        The indices of initial states which will be used as initial states in RIXS calculations.
    temperature: float number
        Temperature (in K) for boltzmann distribution.
    scattering_plane_axis: 3*3 float array
        The local axis defining the scattering plane. The scattering plane is defined in
        the local :math:`zx`-plane.
        local :math:`x`-axis: scattering_plane_axis[:,0]
        local :math:`y`-axis: scattering_plane_axis[:,1]
        local :math:`z`-axis: scattering_plane_axis[:,2]

    Returns
    -------
    rixs: 3d float array
        The calculated RIXS spectra.
    """

    print("edrixs >>> Running RIXS ... ")
    n_ominc = len(ominc_mesh)
    n_eloss = len(eloss_mesh)
    gamma_core = np.zeros(n_ominc, dtype=np.float)
    gamma_final = np.zeros(n_eloss, dtype=np.float)
    if np.isscalar(gamma_c):
        gamma_core[:] = np.ones(n_ominc) * gamma_c
    else:
        gamma_core[:] = gamma_c

    if np.isscalar(gamma_f):
        gamma_final[:] = np.ones(n_eloss) * gamma_f
    else:
        gamma_final[:] = gamma_f

    prob = boltz_dist([eval_i[i] for i in gs_list], temperature)
    rixs = np.zeros((len(ominc_mesh), len(eloss_mesh), len(poltype)), dtype=np.float)
    ntrans, n, m = trans_op.shape
    T_emi = np.zeros((ntrans, m, n), dtype=np.complex128)
    for i in range(ntrans):
        T_emi[i] = np.conj(np.transpose(trans_op[i]))
    polvec_i = np.zeros(ntrans, dtype=np.complex)
    polvec_f = np.zeros(ntrans, dtype=np.complex)

    # Calculate RIXS
    for i, om in enumerate(ominc_mesh):
        F_fi = scattering_mat(eval_i, eval_n, trans_op[:, :, gs_list], T_emi, om, gamma_core[i])
        for j, (it, alpha, jt, beta) in enumerate(poltype):
            ei, ef = dipole_polvec_rixs(thin, thout, phi, alpha, beta,
                                        scattering_plane_axis, (it, jt))
            # dipolar transition
            if ntrans == 3:
                polvec_i[:] = ei
                polvec_f[:] = ef
            # quadrupolar transition
            elif ntrans == 5:
                ki = unit_wavevector(thin, phi, scattering_plane_axis, direction='in')
                kf = unit_wavevector(thout, phi, scattering_plane_axis, direction='out')
                polvec_i[:] = quadrupole_polvec(ei, ki)
                polvec_f[:] = quadrupole_polvec(ef, kf)
            else:
                raise Exception("Have NOT implemented this type of transition operators")
            # scattering magnitude with polarization vectors
            F_mag = np.zeros((len(eval_i), len(gs_list)), dtype=np.complex)
            for m in range(ntrans):
                for n in range(ntrans):
                    F_mag[:, :] += np.conj(polvec_f[m]) * F_fi[m, n] * polvec_i[n]
            for m in gs_list:
                for n in range(len(eval_i)):
                    rixs[i, :, j] += (prob[m] * np.abs(F_mag[n, m])**2 * gamma_final / np.pi /
                                      ((eloss_mesh - (eval_i[n] - eval_i[m]))**2 + gamma_final**2))

    print("edrixs >>> RIXS Done !")
    return rixs


def ed_2v1c(comm, v1_name='f', v2_name='d', c_name='p',
            v1_level=0.0, v2_level=0.0, c_level=0.0,
            v1_soc=(0.0, 0.0), v2_soc=(0.0, 0.0), c_soc=0.0,
            v_tot_noccu=1, slater=([0], [0]),
            v1_ext_B=np.zeros(3), v2_ext_B=np.zeros(3),
            v1_zeeman_on_which='spin', v2_zeeman_on_which='spin',
            v1_cf_mat=None, v2_cf_mat=None,
            v1_other_mat=None, v2_other_mat=None, hopping_v1v2=None,
            ed_solver=2, neval=1, nvector=1, ncv=3, idump=False,
            maxiter=500, eigval_tol=1e-8, min_ndim=1000,
            verbose=0):
    """
    Perform ED for the case of two valence shell plus one Core-shell.
    For example, for U :math:`L_3`-edge RIXS, :math:`2p_{3/2}\\rightarrow 6d` transition,
    the valence shells involved in RIXS are :math:`5f` and :math:`6d`.

    The hopping and Coulomb terms of both the initial and intermediate Hamiltonians will be
    constructed and written to files (hopping_i.in, hopping_n.in, coulomb_i.in and coulomb_n.in).
    Fock basis for the initial Hamiltonian will be written to file (fock_i.in). However,
    ED will only be performed on the initial Hamiltonian to find a few lowest eigenstates.
    Due to large Hilbert space, the ed_fsolver written in Fortran will be called.
    mpi4py and a MPI environment (mpich or openmpi) is required to launch ed_fsolver.

    It will output the eigenvalues in file (eigvals.dat) and eigenvectors in files
    (eigvec.n), where n means the n-th eigenvectors. The files eigvec.n will be used later
    as the inputs for XAS and RIXS calculations.

    Parameters
    ----------
    comm: MPI_comm
        The MPI communicator from mpi4py.
    v1_name: string
        1st valence shell type, can be 's', 'p', 't2g', 'd', 'f'.
    v2_name: string
        2nd valence shell type, can be 's', 'p', 't2g', 'd', 'f'.
    c_name: string
        Core shell type, can be 's', 'p', 'p12', 'p32', 'd', 'd32', 'd52', 'f', 'f52', 'f72'.
    v1_level: float number
        Energy level of 1st valence shell.
    v2_level: float number
        Energy level of 2nd valence shell.
    c_level: float number
        Energy level of core shell.
    v1_soc: tuple of two float numbers
        Spin-orbit coupling strength of 1st valence shell,
        v1_soc[0] for the initial Hamiltonian, and
        v1_soc[1] for the intermediate Hamiltonian.
    v2_soc: tuple of two float numbers
        Spin-orbit coupling strength of 2nd valence shell,
        v2_soc[0] for the initial Hamiltonian, and
        v2_soc[1] for the intermediate Hamiltonian.
    c_soc: float number
        Spin-orbit coupling strength of core electrons.
    v_tot_noccu: int number
        Total number of electrons in both valence shells.
    slater: tuple of two lists
        Slater integrals for initinal (slater[0]) and intermediate (slater[1]) Hamiltonian.
        The order of the elements in the lists should be like this:
        [FX_v1v1, FX_v1v2, GX_v1v2, FX_v2v2, FX_v1c, GX_v1c, FX_v2c, GX_v2c], where
        X are integers with increasing order, it can be X=0, 2, 4, 6 or X=1, 3, 5.
    v1_ext_B: list of three float numbers
        Vector of external magnetic field on 1st valence shell,
        with respect to global :math:`xyz`-axis.
    v2_ext_B: list of three float numbers
        Vector of external magnetic field on 2nd valence shell,
        with respect to global :math:`xyz`-axis.
    v1_zeeman_on_which: string
        Apply Zeeman exchange field on 'spin', 'orbital' or 'both', for 1st valence shell.
    v2_zeeman_on_which: string
        Apply Zeeman exchange field on 'spin', 'orbital' or 'both', for 2nd valence shell.
    v1_cf_mat: 2d complex array
        Crystal field splitting Hamiltonian of 1st valence shell in the complex harmonics
        basis with spin degree of freedom. The orbital order is:
        :math:`|-l,\\uparrow>, |-l,\\downarrow>, |-l+1,\\uparrow>, |-l+1,\\downarrow>, ...,
        |l,\\uparrow>, |l,\\downarrow>`.
    v2_cf_mat: 2d complex array
        Crystal field splitting Hamiltonian of 2nd valence shell in the complex harmonics
        basis with spin degree of freedom. The orbital order is:
        :math:`|-l,\\uparrow>, |-l,\\downarrow>, |-l+1,\\uparrow>, |-l+1,\\downarrow>, ...,
        |l,\\uparrow>, |l,\\downarrow>`.
    v1_other_mat: 2d complex array
        Other Hamiltonian of 1st valence shell in the complex harmonics
        basis with spin degree of freedom. The orbital order is:
        :math:`|-l,\\uparrow>, |-l,\\downarrow>, |-l+1,\\uparrow>, |-l+1,\\downarrow>, ...,
        |l,\\uparrow>, |l,\\downarrow>`.
    v2_other_mat: 2d complex array
        Other Hamiltonian of 2nd valence shell in the complex harmonics
        basis with spin degree of freedom. The orbital order is:
        :math:`|-l,\\uparrow>, |-l,\\downarrow>, |-l+1,\\uparrow>, |-l+1,\\downarrow>, ...,
        |l,\\uparrow>, |l,\\downarrow>`.
    hopping_v1v2: 2d complex array
        Hopping between the two valence shells. The 1st index is for v1 and 2nd index is for v2.
    ed_solver: int
        Type of ED solver, options can be 0, 1, 2

        0: use Lapack to fully diagonalize Hamiltonian to get all the eigenvalues.

        1: use standard Lanczos algorithm to find only a few lowest eigenvalues,
        no re-orthogonalization has been applied, so it is not very accurate.

        2: use parallel version of Arpack library to find a few lowest eigenvalues,
        it is accurate, so it is the recommeded choice in real calculations of XAS and RIXS.
    neval: int
        Number of eigenvalues to be found. For ed_solver=2, the value should not be too small,
        neval > 10 is usually a safe value.
    nvector: int
        Number of eigenvectors to be found.
    ncv: int
        Used for ed_solver=2, it should be at least ncv > neval + 2. Usually, set it a little
        bit larger than neval, for example, ncv=200, when neval=100.
    idump: logical
        Whether to dump the eigenvectors to files "eigvec.n", where n means the n-th vectors.
    maxiter: int
        Maximum number of iterations in finding all the eigenvalues, used for ed_solver=1,2.
    eigval_tol: float
        The convergence criteria of eigenvalues, used for ed_solver=1,2.
    min_ndim: int
        The minimum dimension of the Hamiltonian when the ed_solver=1,2 can be used.
    verbose: int
        Level of dumping files.

    Returns
    -------
    v1v2_norb: int
        Number of total valence orbitals.
    c_norb: int
        Number of core orbitals. 
    emat_i: 2d complex array
        Hopping terms of initial Hamiltonian.
    emat_n: 2d complex array
        Hopping terms of intermediate Hamiltonian.
    umat_i: 4d complex array
        Coulomb terms of initial Hamiltonian.
    umat_n: 4d complex array
        Coulomb terms of intermediate Hamiltonian.
    eval_i: 1d float array
        The eigenvalues of initial Hamiltonian.
    denmat: 2d complex array
        The density matrix in the eigenstates.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    fcomm = comm.py2f()
    if rank == 0:
        print("edrixs >>> Running ED ...", flush=True)

    v1_name = v1_name.strip()
    v2_name = v2_name.strip()
    c_name = c_name.strip()
    info_shell = info_atomic_shell()
    # Quantum numbers of angular momentum
    v1_orbl = info_shell[v1_name][0]
    v2_orbl = info_shell[v2_name][0]

    # number of orbitals with spin
    v1_norb = info_shell[v1_name][1]
    v2_norb = info_shell[v2_name][1]
    c_norb = info_shell[c_name][1]
    # total number of orbitals
    ntot = v1_norb + v2_norb + c_norb
    v1v2_norb = v1_norb + v2_norb

    # Coulomb interaction
    slater_name = slater_integrals_name((v1_name, v2_name, c_name), ('v1', 'v2', 'c1'))
    num_slat = len(slater_name)
    slater_i = np.zeros(num_slat, dtype=np.float)
    slater_n = np.zeros(num_slat, dtype=np.float)
    if num_slat > len(slater[0]):
        slater_i[0:len(slater[0])] = slater[0]
    else:
        slater_i[:] = slater[0][0:num_slat]
    if num_slat > len(slater[1]):
        slater_n[0:len(slater[1])] = slater[1]
    else:
        slater_n[:] = slater[1][0:num_slat]

    # print summary of slater integrals
    if rank == 0:
        print("    Summary of Slater integrals:", flush=True)
        print("    ------------------------------", flush=True)
        print("    Terms,  Initial,  Intermediate", flush=True)
        for i in range(num_slat):
            print(
                "    ", slater_name[i],
                ":  {:20.10f}{:20.10f}".format(slater_i[i], slater_n[i]), flush=True
            )

    umat_i = get_umat_slater_3shells((v1_name, v2_name, c_name), *slater_i)
    umat_n = get_umat_slater_3shells((v1_name, v2_name, c_name), *slater_n)

    if rank == 0:
        write_umat(umat_i, 'coulomb_i.in')
        write_umat(umat_n, 'coulomb_n.in')

    emat_i = np.zeros((ntot, ntot), dtype=np.complex)
    emat_n = np.zeros((ntot, ntot), dtype=np.complex)
    # SOC
    if v1_name in ['p', 'd', 't2g', 'f']:
        emat_i[0:v1_norb, 0:v1_norb] += atom_hsoc(v1_name, v1_soc[0])
        emat_n[0:v1_norb, 0:v1_norb] += atom_hsoc(v1_name, v1_soc[1])
    if v2_name in ['p', 'd', 't2g', 'f']:
        emat_i[v1_norb:v1v2_norb, v1_norb:v1v2_norb] += atom_hsoc(v2_name, v2_soc[0])
        emat_n[v1_norb:v1v2_norb, v1_norb:v1v2_norb] += atom_hsoc(v2_name, v2_soc[1])
    if c_name in ['p', 'd', 'f']:
        emat_n[v1v2_norb:ntot, v1v2_norb:ntot] += atom_hsoc(c_name, c_soc)

    # crystal field
    if v1_cf_mat is not None:
        emat_i[0:v1_norb, 0:v1_norb] += v1_cf_mat
        emat_n[0:v1_norb, 0:v1_norb] += v1_cf_mat
    if v2_cf_mat is not None:
        emat_i[v1_norb:v1v2_norb, v1_norb:v1v2_norb] += v2_cf_mat
        emat_n[v1_norb:v1v2_norb, v1_norb:v1v2_norb] += v2_cf_mat
    # other mat
    if v1_other_mat is not None:
        emat_i[0:v1_norb, 0:v1_norb] += v1_other_mat
        emat_n[0:v1_norb, 0:v1_norb] += v1_other_mat
    if v2_other_mat is not None:
        emat_i[v1_norb:v1v2_norb, v1_norb:v1v2_norb] += v2_other_mat
        emat_n[v1_norb:v1v2_norb, v1_norb:v1v2_norb] += v2_other_mat

    # energy of shell
    emat_i[0:v1_norb, 0:v1_norb] += np.eye(v1_norb) * v1_level
    emat_n[0:v1_norb, 0:v1_norb] += np.eye(v1_norb) * v1_level
    emat_i[v1_norb:v1v2_norb, v1_norb:v1v2_norb] += np.eye(v2_norb) * v2_level
    emat_n[v1_norb:v1v2_norb, v1_norb:v1v2_norb] += np.eye(v2_norb) * v2_level
    emat_i[v1v2_norb:ntot, v1v2_norb:ntot] += np.eye(c_norb) * c_level
    emat_n[v1v2_norb:ntot, v1v2_norb:ntot] += np.eye(c_norb) * c_level

    # external magnetic field
    for name, l, ext_B, which, i1, i2 in [
        (v1_name, v1_orbl, v1_ext_B, v1_zeeman_on_which, 0, v1_norb),
        (v2_name, v2_orbl, v2_ext_B, v2_zeeman_on_which, v1_norb, v1v2_norb)
    ]:
        if name == 't2g':
            lx, ly, lz = get_lx(1, True), get_ly(1, True), get_lz(1, True)
            sx, sy, sz = get_sx(1), get_sy(1), get_sz(1)
            lx, ly, lz = -lx, -ly, -lz
        else:
            lx, ly, lz = get_lx(l, True), get_ly(l, True), get_lz(l, True)
            sx, sy, sz = get_sx(l), get_sy(l), get_sz(l)
        if which.strip() == 'spin':
            zeeman = ext_B[0] * (2 * sx) + ext_B[1] * (2 * sy) + ext_B[2] * (2 * sz)
        elif which.strip() == 'orbital':
            zeeman = ext_B[0] * lx + ext_B[1] * ly + ext_B[2] * lz
        elif which.strip() == 'both':
            zeeman = (ext_B[0] * (lx + 2 * sx) +
                      ext_B[1] * (ly + 2 * sy) +
                      ext_B[2] * (lz + 2 * sz))
        else:
            raise Exception("Unknown value of zeeman_on_which", which)
        emat_i[i1:i2, i1:i2] += zeeman
        emat_n[i1:i2, i1:i2] += zeeman

    # hopping between the two valence shells
    if hopping_v1v2 is not None:
        emat_i[0:v1_norb, v1_norb:v1v2_norb] += hopping_v1v2
        emat_i[v1_norb:v1v2_norb, 0:v1_norb] += np.conj(np.transpose(hopping_v1v2))
        emat_n[0:v1_norb, v1_norb:v1v2_norb] += hopping_v1v2
        emat_n[v1_norb:v1v2_norb, 0:v1_norb] += np.conj(np.transpose(hopping_v1v2))

    if rank == 0:
        write_emat(emat_i, 'hopping_i.in')
        write_emat(emat_n, 'hopping_n.in')
        write_config(
            './', ed_solver, v1v2_norb, c_norb, neval, nvector, ncv, idump,
            maxiter=maxiter, min_ndim=min_ndim, eigval_tol=eigval_tol
        )
        write_fock_dec_by_N(v1v2_norb, v_tot_noccu, "fock_i.in")

    # now, call ed solver
    comm.Barrier()
    ed_fsolver(fcomm, rank, size)
    comm.Barrier()

    # read eigvals.dat and denmat.dat
    data = np.loadtxt('eigvals.dat')
    eval_i = np.zeros(neval, dtype=np.float)
    eval_i[0:neval] = data[0:neval, 1]
    data = np.loadtxt('denmat.dat')
    tmp = (nvector, v1v2_norb, v1v2_norb)
    denmat = data[:, 3].reshape(tmp) + 1j * data[:, 4].reshape(tmp)

    return v1v2_norb, c_norb, emat_i, emat_n, umat_i, umat_n, eval_i, denmat 


def xas_2v1c(comm, ominc_mesh, gamma_c=0.1,
             v1_name='f', v2_name='d', c_name='p',
             v_tot_noccu=1, trans_to_which=1, thin=1.0, phi=0,
             poltype=[('isotropic', 0.0)],
             num_gs=1, temperature=1.0, local_axis=np.eye(3),
             scattering_plane_axis=np.eye(3), nkryl=200):
    """
    Calculate XAS for the case with 2 valence shells plus 1 core shell.

    Parameters
    ----------
    comm: MPI_comm
        MPI communicator.
    v1_name: string
        1st valence shell type, can be 's', 'p', 't2g', 'd', 'f'.
    v2_name: string
        2nd valence shell type, can be 's', 'p', 't2g', 'd', 'f'.
    c_name: string
        Core shell type, can be 's', 'p', 'p12', 'p32', 'd', 'd32', 'd52', 'f', 'f52', 'f72'.
    v_tot_noccu: int
        Total occupancy of valence shells.
    ominc_mesh: 1d float array
        The mesh of the incident energy of photon.
    gamma_c: a float number or a 1d float array with same length as ominc_mesh.
        The core-hole life-time broadening factor. It can be a constant value
        or incident energy dependent.
    trans_to_which: int
        Photon transition to which valence shell.

        1: to 1st valence shell,

        2: to 2nd valence shell.

    thin: float number
        The incident angle of photon.
    phi: float number
        Azimuthal angle in radian, defined with respect to the
        :math:`x`-axis: scattering_plane_axis[:,0].
    poltype: list of tuples
        Type of polarization, options are
        ('linear', alpha), where alpha is the angle between the polarization vector and
        the scattering plane. ('left', 0), ('right', 0), ('isotropic', 0).
    num_gs: int
        Number of initial states used in XAS calculations.
    temperature: float number
        Temperature (in K) for boltzmann distribution.
    local_axis: 3*3 float array
        The local axis defining the orbitals.
    scattering_plane_axis: 3*3 float array
        The local axis defining the scattering plane. The scattering plane is defined in
        the local :math:`zx`-plane.
        local :math:`x`-axis: scattering_plane_axis[:,0]
        local :math:`y`-axis: scattering_plane_axis[:,1]
        local :math:`z`-axis: scattering_plane_axis[:,2]
    nkryl: int
        Maximum of poles obtained.

    Returns
    -------
    xas: 2d array
        The calculated XAS spectra.
    poles: list of dict
        The calculated XAS poles.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    fcomm = comm.py2f()

    v1_name = v1_name.strip()
    v2_name = v2_name.strip()
    c_name = c_name.strip()
    info_shell = info_atomic_shell()
    v1_norb = info_shell[v1_name][1]
    v2_norb = info_shell[v2_name][1]
    c_norb = info_shell[c_name][1]
    ntot = v1_norb + v2_norb + c_norb
    v1v2_norb = v1_norb + v2_norb
    if rank == 0:
        print("edrixs >>> Running XAS ...", flush=True)
        write_config(num_val_orbs=v1v2_norb, num_core_orbs=c_norb,
                     num_gs=num_gs, nkryl=nkryl)
        write_fock_dec_by_N(v1v2_norb, v_tot_noccu, "fock_i.in")
        write_fock_dec_by_N(v1v2_norb, v_tot_noccu + 1, "fock_n.in")

        # Build transition operators in local-xyz axis
        if trans_to_which == 1:
            case = v1_name + c_name
        elif trans_to_which == 2:
            case = v2_name + c_name
        else:
            raise Exception('Unkonwn trans_to_which: ', trans_to_which)
        tmp = get_trans_oper(case)
        npol, n, m = tmp.shape
        tmp_g = np.zeros((npol, n, m), dtype=np.complex)
        trans_mat = np.zeros((npol, ntot, ntot), dtype=np.complex)
        # Transform the transition operators to global-xyz axis
        # dipolar transition
        if npol == 3:
            for i in range(3):
                for j in range(3):
                    tmp_g[i] += local_axis[i, j] * tmp[j]
        # quadrupolar transition
        elif npol == 5:
            alpha, beta, gamma = rmat_to_euler(local_axis)
            wignerD = get_wigner_dmat(4, alpha, beta, gamma)
            rotmat = np.dot(np.dot(tmat_r2c('d'), wignerD), np.conj(np.transpose(tmat_r2c('d'))))
            for i in range(5):
                for j in range(5):
                    tmp_g[i] += rotmat[i, j] * tmp[j]
        else:
            raise Exception("Have NOT implemented this case: ", npol)
        if trans_to_which == 1:
            trans_mat[:, 0:v1_norb, v1v2_norb:ntot] = tmp_g    
        else:
            trans_mat[:, v1_norb:v1v2_norb, v1v2_norb:ntot] = tmp_g    


    n_om = len(ominc_mesh)
    gamma_core = np.zeros(n_om, dtype=np.float)
    if np.isscalar(gamma_c):
        gamma_core[:] = np.ones(n_om) * gamma_c
    else:
        gamma_core[:] = gamma_c

    # loop over different polarization
    xas = np.zeros((n_om, len(poltype)), dtype=np.float)
    poles = []
    comm.Barrier()
    for it, (pt, alpha) in enumerate(poltype):
        if rank == 0:
            kvec = unit_wavevector(thin, phi, scattering_plane_axis, 'in')
            polvec = np.zeros(npol, dtype=np.complex)
            if pt.strip() == 'isotropic':
                pol = np.ones(3)/np.sqrt(3.0)
            elif pt.strip() == 'left' or pt.strip() == 'right' or pt.strip() == 'linear':
                pol = dipole_polvec_xas(thin, phi, alpha, scattering_plane_axis, pt)
            else:
                raise Exception("Unknown polarization type: ", pt)
            if npol == 3:  # Dipolar transition
                polvec[:] = pol
            if npol == 5:  # Quadrupolar transition
                polvec[:] = quadrupole_polvec(pol, kvec)

            trans = np.zeros((ntot, ntot), dtype=np.complex)
            for i in range(npol):
                trans[:, :] += trans_mat[i] * polvec[i]
            write_emat(trans, 'transop_xas.in')

        # call XAS solver in fedrixs
        comm.Barrier()
        xas_fsolver(fcomm, rank, size)
        comm.Barrier()

        file_list = ['xas_poles.' + str(i+1) for i in range(num_gs)]
        pole_dict = read_poles_from_file(file_list)
        poles.append(pole_dict)
        xas[:, it] = get_spectra_from_poles(pole_dict, ominc_mesh, gamma_core, temperature)

        # rename pole files 
        if rank == 0:
            postfix = '_' + 'pol' + str(it+1)
            rename_pole_file(file_list, postfix)

    return xas, poles
