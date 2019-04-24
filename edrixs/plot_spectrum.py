__all__ = ['get_spectra_from_poles', 'plot_spectrum', 'plot_rixs_map']

import numpy as np
import matplotlib.pyplot as plt
from .utils import boltz_dist


def get_spectra_from_poles(poles_dict, omega_mesh, gamma_mesh, temperature):
    nom = len(omega_mesh)
    spectra = np.zeros(nom, dtype=np.float64)
    gs_dist = boltz_dist(poles_dict['eigval'], temperature)
    ngs = len(poles_dict['eigval'])
    for i in range(ngs):
        tmp_vec = np.zeros(nom, dtype=np.complex)
        neff = poles_dict['npoles'][i]
        alpha = poles_dict['alpha'][i]
        beta = poles_dict['beta'][i]
        eigval = poles_dict['eigval'][i]
        norm = poles_dict['norm'][i]
        for j in range(neff - 1, 0, -1):
            tmp_vec = beta[j - 1]**2 / (omega_mesh + 1j * gamma_mesh + eigval
                                        - alpha[j] - tmp_vec)
        tmp_vec = 1.0 / (omega_mesh + 1j * gamma_mesh +
                         eigval - alpha[0] - tmp_vec)
        spectra[:] += -1.0 / np.pi * np.imag(tmp_vec) * norm * gs_dist[i]

    return spectra


def plot_spectrum(file_list, omega_mesh, gamma_mesh, T=1.0, fname='spectrum.dat',
                  om_shift=0.0, fmt_float='{:.15f}'):
    """
    Reading poles :math:`\\alpha` and :math:`\\beta`, and calculate
    the spectrum using continued fraction formula,

    .. math::
        I(\\omega_{i}) =-\\frac{1}{\\pi}\\text{Im} \\left[ \\frac{1}{x - \\alpha_{0} -
        \\frac{\\beta_{1}^2}{x-\\alpha_{1} - \\frac{\\beta_{2}^2}{x-\\alpha_{2} - ...}} }\\right],

    where, :math:`x = \\omega_{i}+i\\Gamma_{i} + E_{g}`.

    Parameters
    ----------
    file_list: list of string
        Name of poles file.
    omega_mesh: 1d float array
        The frequency mesh.
    gamma_mesh: 1d float array
        The broadening factor, in general, it is frequency dependent.
    T: float (default: 1.0K)
        Temperature (K).
    fname: str (default: 'spectrum.dat')
        File name to store spectrum.
    om_shift: float (default: 0.0)
        Energy shift.
    fmt_float: str (default: '{:.15f}')
        Format for printing float numbers.
    """

    ngs = len(file_list)
    eigvals = np.zeros(ngs, dtype=np.float64)
    norm = np.zeros(ngs, dtype=np.float64)
    neff = np.zeros(ngs, dtype=np.int)

    alpha = []
    beta = []
    for i, igs in enumerate(file_list):
        f = open(igs, 'r')

        line = f.readline()
        neff[i] = int(line.strip().split()[1])

        line = f.readline()
        eigvals[i] = float(line.strip().split()[1])

        line = f.readline()
        norm[i] = float(line.strip().split()[1])

        tmp_a = []
        tmp_b = []
        for j in range(neff[i]):
            line = f.readline()
            line = line.strip().split()
            tmp_a.append(float(line[1]))
            tmp_b.append(float(line[2]))

        alpha.append(tmp_a)
        beta.append(tmp_b)

        f.close()

    nom_inc = len(omega_mesh)
    spectrum = np.zeros(nom_inc, dtype=np.float64)
    gs_dist = boltz_dist(eigvals, T)
    print("Probabilities of initial states: ", gs_dist)

    for i in range(ngs):
        tmp_vec = np.zeros(nom_inc, dtype=np.complex128)
        for j in range(neff[i] - 1, 0, -1):
            tmp_vec = beta[i][j - 1]**2 / (omega_mesh + 1j * gamma_mesh + eigvals[i]
                                           - alpha[i][j] - tmp_vec)

        tmp_vec = 1.0 / (omega_mesh + 1j * gamma_mesh +
                         eigvals[i] - alpha[i][0] - tmp_vec)
        spectrum[:] += -1.0 / np.pi * np.imag(tmp_vec) * norm[i] * gs_dist[i]

    space = "  "
    fmt_string = (fmt_float + space) * 2 + '\n'
    f = open(fname, 'w')
    for i in range(nom_inc):
        f.write(fmt_string.format(omega_mesh[i] + om_shift, spectrum[i]))
    f.close()


def plot_rixs_map(rixs_data, ominc_mesh, eloss_mesh, fname='rixs_map.pdf'):
    """
    Given 2d RIXS data, plot a RIXS map and save it to a pdf file.

    Parameters
    ----------
    rixs_data: 2d float array
        Calculated RIXS data as a function of incident energy and energy loss.
    ominc_mesh: 1d float array
        Incident energy mesh.
    eloss_mesh: 1d float array
        Energy loss mesh.
    fname: string
        File name to save RIXS map.
    """

    fig, ax = plt.subplots()
    m, n = np.array(rixs_data).shape
    if len(ominc_mesh) == m and len(eloss_mesh) == n:
        plt.imshow(
            rixs_data, extent=[min(eloss_mesh), max(eloss_mesh), min(ominc_mesh), max(ominc_mesh)],
            origin='lower', aspect='auto', cmap='rainbow', interpolation='gaussian')
        plt.xlabel(r'Energy loss (eV)')
        plt.ylabel(r'Energy of incident photon (eV)')
    elif len(eloss_mesh) == m and len(ominc_mesh) == n:
        plt.imshow(
            rixs_data, extent=[min(ominc_mesh), max(ominc_mesh), min(eloss_mesh), max(eloss_mesh)],
            origin='lower', aspect='auto', cmap='rainbow', interpolation='gaussian')
        plt.ylabel(r'Energy loss (eV)')
        plt.xlabel(r'Energy of incident photon (eV)')
    else:
        raise Exception("Dimension of rixs_data is not consistent with ominc_mesh or eloss_mesh")
    plt.savefig(fname)
