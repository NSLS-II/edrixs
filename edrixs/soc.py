__all__ = ['atom_hsoc']

import numpy as np


def atom_hsoc(case, soc):
    """
    Return atomic spin-orbit coupling matrix :math:`\\vec{l}\\cdot\\vec{s}`
    in complex spherical harmonics basis.

    Parameters
    ----------
    case: str
        String label indicating atomic shell,

        - 'p': for :math:`p` -shell.
        - 't2g': for :math:`t_{2g}` -shell.
        - 'd': for :math:`d` -shell.
        - 'f': for :math:`f` -shell.
    soc: float
        The strength of spin-orbit coupling.

    Returns
    -------
    hsoc: 2d complex array
        The spin-orbit coupling matrix.
    """

    sqrt2 = np.sqrt(2.0)
    sqrt6 = np.sqrt(6.0)
    sqrt10 = np.sqrt(10.0)
    sqrt12 = np.sqrt(12.0)

    if case.strip() == 'p':
        hsoc = np.zeros((6, 6), dtype=np.complex128)
        hsoc[0, 0] = -1.0
        hsoc[3, 0] = sqrt2
        hsoc[1, 1] = 1.0
        hsoc[5, 2] = sqrt2
        hsoc[0, 3] = sqrt2
        hsoc[4, 4] = 1.0
        hsoc[2, 5] = sqrt2
        hsoc[5, 5] = -1.0
        return 0.5 * soc * hsoc

    elif case.strip() == 't2g':
        hsoc = np.zeros((6, 6), dtype=np.complex128)
        hsoc[0, 0] = -1.0
        hsoc[3, 0] = sqrt2
        hsoc[1, 1] = 1.0
        hsoc[5, 2] = sqrt2
        hsoc[0, 3] = sqrt2
        hsoc[4, 4] = 1.0
        hsoc[2, 5] = sqrt2
        hsoc[5, 5] = -1.0
        return 0.5 * -soc * hsoc

    elif case.strip() == 'd':
        hsoc = np.zeros((10, 10), dtype=np.complex128)
        hsoc[0, 0] = -2.0
        hsoc[3, 0] = 2.0
        hsoc[1, 1] = 2.0
        hsoc[2, 2] = -1.0
        hsoc[5, 2] = sqrt6
        hsoc[0, 3] = 2.0
        hsoc[3, 3] = 1.0
        hsoc[7, 4] = sqrt6
        hsoc[2, 5] = sqrt6
        hsoc[6, 6] = 1.0
        hsoc[9, 6] = 2.0
        hsoc[4, 7] = sqrt6
        hsoc[7, 7] = -1.0
        hsoc[8, 8] = 2.0
        hsoc[6, 9] = 2.0
        hsoc[9, 9] = -2.0
        return 0.5 * soc * hsoc

    elif case.strip() == 'f':
        hsoc = np.zeros((14, 14), dtype=np.complex128)
        hsoc[0, 0] = -3.0
        hsoc[3, 0] = sqrt6
        hsoc[1, 1] = 3.0
        hsoc[2, 2] = -2.0
        hsoc[5, 2] = sqrt10
        hsoc[0, 3] = sqrt6
        hsoc[3, 3] = 2.0
        hsoc[4, 4] = -1.0
        hsoc[7, 4] = sqrt12
        hsoc[2, 5] = sqrt10
        hsoc[5, 5] = 1.0
        hsoc[9, 6] = sqrt12
        hsoc[4, 7] = sqrt12
        hsoc[8, 8] = 1.0
        hsoc[11, 8] = sqrt10
        hsoc[6, 9] = sqrt12
        hsoc[9, 9] = -1.0
        hsoc[10, 10] = 2.0
        hsoc[13, 10] = sqrt6
        hsoc[8, 11] = sqrt10
        hsoc[11, 11] = -2.0
        hsoc[12, 12] = 3.0
        hsoc[10, 13] = sqrt6
        hsoc[13, 13] = -3.0
        return 0.5 * soc * hsoc

    else:
        raise Exception("error in atom_hsoc: DO NOT support SOC for this case: ", case)
