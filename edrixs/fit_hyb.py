__all__ = ['fit_func', 'fit_hyb', 'get_hyb']

import numpy as np
from scipy.optimize import curve_fit


def fit_func(x, *args):
    """
    Given frequency :math:`\\omega`, bath energy level :math:`\\epsilon_{l}` and
    the hybridization strength :math:`V_{l}`,
    return the hybridization function,

    .. math::

        \\Delta(\\omega)=\\sum_{l=1}^{N}\\frac{|V_{l}|^2}{\\omega-\\epsilon_{l}}.

    Parameters
    ----------
    x: 1d float array
        Frequency :math:`\\omega`, the first half is the real part and
        the second half is the imaginary part.
    args: 1d float array
        The first half is the bath energy level :math:`\\epsilon_{l}` and the
        second half if the hybridization strength :math:`V_{l}`.

    Returns
    -------
    y: 1d float array
        The calculated hybridization function :math:`\\Delta(\\omega)`, the
        first half is the real part and the second half is the imaginary part.
    """

    m = len(x) // 2
    n = len(args) // 2
    y = np.zeros(len(x), dtype=np.float64)

    tmp_x = np.zeros(m, dtype=np.complex128)
    tmp_y = np.zeros(m, dtype=np.complex128)
    tmp_x[:] = x[0:m] + 1j * x[m:2 * m]

    for i in range(n):
        tmp_y[:] += args[n + i]**2 / (tmp_x[:] - args[i])

    y[0:m] = tmp_y.real
    y[m:2 * m] = tmp_y.imag

    return y


def fit_hyb(x, y, N, p0):
    """
    Given the hybridization function :math:`\\Delta(\\omega)`,
    call function curve_fit in scipy to
    fit bath energy levels :math:`\\epsilon_{l}` and
    hybridization strength :math:`V_{l}`.

    .. math::

        \\Delta(\\omega)=\\sum_{l=1}^{N}\\frac{|V_{l}|^2}{\\omega-\\epsilon_{l}}.


    Parameters
    ----------
    x: 1d complex array
       Frequency :math:`\\omega`.
    y: 1d complex array
       Hybridization function :math:`\\Delta(\\omega)`.
    N: int
        Number of bath sites
    p0: N-length 1d float array
        Initial guess, the first half is :math:`\\epsilon_{l}` and
        the second half is :math:`V_{l}`.

    Returns
    -------
    e: N-length 1d float array
        The fitted bath energy levels :math:`\\epsilon_{l}`.
    v: N-length 1d float array
        The fitted hybridization strength :math:`V_{l}`.
    """

    m = len(x)
    xdata = np.zeros(2 * m, dtype=np.float64)
    ydata = np.zeros(2 * m, dtype=np.float64)
    xdata[0:m], xdata[m:2 * m] = x.real, x.imag
    ydata[0:m], ydata[m:2 * m] = y.real, y.imag
    popt, pcov = curve_fit(fit_func, xdata, ydata, p0)
    e, v = popt[0:N], popt[N:2 * N]
    return e, v


def get_hyb(x, e, v):
    """
    Given the fitted :math:`\\epsilon_{l}` and :math:`V_{l}`, calcualte the
    hybridization function :math:`\\Delta(\\omega)`,

    .. math::

        \\Delta(\\omega)=\\sum_{l=1}^{N}\\frac{|V_{l}|^2}{\\omega-\\epsilon_{l}}.

    Parameters
    ----------
    x: 1d complex array
       Frequency :math:`\\omega`.
    e:  N-length 1d float array
       The fitted bath energy levels.
    v:  N-length 1d float array
        The fitted hybridization strength.

    Returns
    -------
    y: 1d complex array
        The calculated hybridization function :math:`\\Delta(\\omega)`.
    """

    y = np.zeros(len(x), dtype=np.complex128)
    for i in range(len(e)):
        y[:] += v[i]**2 / (x[:] - e[i])
    return y
