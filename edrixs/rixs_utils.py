__all__ = ['scattering_mat']

import numpy as np


def scattering_mat(eval_i, eval_n, trans_mat_abs,
                   trans_mat_emi, omega_inc, gamma_n):
    """
    Calculate X-ray scattering magnitude.

    .. math::

        F^{ab}_{fi} = \\sum_{n}\\frac{<f|T_{a}|n><n|T_{b}|i>}{\\omega_{in}
                    - E_{n} + E_{i} + i\\Gamma_{n}},

    where, :math:`T_{a}` and :math:`T_{b}` are components of transition
    operators( :math:`a,b=x,y,z`).

    Parameters
    ----------
    eval_i: 1d float array
        Eigenvalues of the initial configuration without core-hole, :math:`E_i`.
    eval_n: 1d float array
        Eigenvalues of the intermediate configuration with core-hole, :math:`E_n`.
    trans_mat_abs: 3d complex array
        The transition operator for absorption process, :math:`<n|T_{b}|i>`.
    trans_mat_emi: 3d complex array
        The transition operator for emission process, :math:`<f|T_{a}|n>`.
    omega_inc: float
        The energy of incident photon, :math:`\\omega_{in}`.
    gamma_n: float
        The broadening of the core-hole (eV), :math:`\\Gamma_{n}`.

    Returns
    -------
    Ffi: 4d complex array
        The calculated scattering magnitude, :math:`F^{ab}_{fi}`.
    """

    num_gs = trans_mat_abs.shape[2]
    num_ex = trans_mat_abs.shape[1]
    num_fs = trans_mat_emi.shape[1]

    npol_abs = trans_mat_abs.shape[0]
    npol_emi = trans_mat_emi.shape[0]

    Ffi = np.zeros((npol_emi, npol_abs, num_fs, num_gs), dtype=np.complex128)
    tmp_abs = np.zeros((npol_abs, num_ex, num_gs), dtype=np.complex128)
    denomi = np.zeros((num_ex, num_gs), dtype=np.complex128)

    for i in range(num_ex):
        for j in range(num_gs):
            aa = omega_inc - (eval_n[i] - eval_i[j])
            denomi[i, j] = 1.0 / (aa + 1j * gamma_n)

    for i in range(npol_abs):
        tmp_abs[i] = trans_mat_abs[i] * denomi

    for i in range(npol_emi):
        for j in range(npol_abs):
            Ffi[i, j, :, :] = np.dot(trans_mat_emi[i], tmp_abs[j])

    return Ffi
