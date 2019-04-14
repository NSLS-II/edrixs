#!/usr/bin/env python
import sys
import numpy as np


def write_tensor_1(tensor, fname, only_nonzeros=True, tol=1E-10, fmt_int='{:10d}',
                   fmt_float='{:.15f}'):
    (n1, ) = tensor.shape
    is_cmplx = False
    if tensor.dtype == np.complex or tensor.dtype == np.complex128:
        is_cmplx = True
    space = "    "
    f = open(fname, 'w')
    for i in range(n1):
        if only_nonzeros and abs(tensor[i]) < tol:
            continue
        if is_cmplx:
            fmt_string = fmt_int + space + (fmt_float + space) * 2 + '\n'
            f.write(fmt_string.format(i + 1, tensor[i].real, tensor[i].imag))
        else:
            fmt_string = fmt_int + space + fmt_float + space + '\n'
            f.write(fmt_string.format(i + 1, tensor[i]))
    f.close()


def write_tensor_2(tensor, fname, only_nonzeros=True, tol=1E-10, fmt_int='{:10d}',
                   fmt_float='{:.15f}'):
    (n1, n2) = tensor.shape
    is_cmplx = False
    if tensor.dtype == np.complex or tensor.dtype == np.complex128:
        is_cmplx = True
    space = "    "
    f = open(fname, 'w')
    for i in range(n1):
        for j in range(n2):
            if only_nonzeros and abs(tensor[i, j]) < tol:
                continue
            if is_cmplx:
                fmt_string = (fmt_int + space) * 2 + \
                    (fmt_float + space) * 2 + '\n'
                f.write(fmt_string.format(i + 1, j + 1,
                                          tensor[i, j].real, tensor[i, j].imag))
            else:
                fmt_string = (fmt_int + space) * 2 + fmt_float + space + '\n'
                f.write(fmt_string.format(i + 1, j + 1, tensor[i, j]))
    f.close()


def write_tensor_3(tensor, fname, only_nonzeros=True, tol=1E-10, fmt_int='{:10d}',
                   fmt_float='{:.15f}'):
    (n1, n2, n3) = tensor.shape
    is_cmplx = False
    if tensor.dtype == np.complex or tensor.dtype == np.complex128:
        is_cmplx = True
    space = "    "
    f = open(fname, 'w')
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                if only_nonzeros and abs(tensor[i, j, k]) < tol:
                    continue
                if is_cmplx:
                    fmt_string = (fmt_int + space) * 3 + \
                        (fmt_float + space) * 2 + '\n'
                    f.write(fmt_string.format(i + 1, j + 1, k + 1, tensor[i, j, k].real,
                                              tensor[i, j, k].imag))
                else:
                    fmt_string = (fmt_int + space) * 3 + \
                        fmt_float + space + '\n'
                    f.write("{:10d}{:10d}{:10d}{:20.12E}\n".format(i + 1, j + 1, k + 1,
                                                                   tensor[i, j, k]))
    f.close()


def write_tensor_4(tensor, fname, only_nonzeros=True, tol=1E-10, fmt_int='{:10d}',
                   fmt_float='{:.15f}'):
    (n1, n2, n3, n4) = tensor.shape
    is_cmplx = False
    if tensor.dtype == np.complex or tensor.dtype == np.complex128:
        is_cmplx = True
    space = "    "
    f = open(fname, 'w')
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                for l in range(n4):
                    if only_nonzeros and abs(tensor[i, j, k, l]) < tol:
                        continue
                    if is_cmplx:
                        fmt_string = (fmt_int + space) * 4 + \
                            (fmt_float + space) * 2 + '\n'
                        f.write(fmt_string.format(i + 1, j + 1, k + 1, l + 1, tensor[i, j, k, l].real,
                                                  tensor[i, j, k, l].imag))
                    else:
                        fmt_string = (fmt_int + space) * 4 + \
                            fmt_float + space + '\n'
                        f.write(fmt_string.format(i + 1, j + 1,
                                                  k + 1, l + 1, tensor[i, j, k, l]))
    f.close()


def write_tensor_5(tensor, fname, only_nonzeros=True, tol=1E-10, fmt_int='{:10d}',
                   fmt_float='{:.15f}'):
    (n1, n2, n3, n4, n5) = tensor.shape
    is_cmplx = False
    if tensor.dtype == np.complex or tensor.dtype == np.complex128:
        is_cmplx = True
    space = "    "
    f = open(fname, 'w')
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                for l in range(n4):
                    for m in range(n5):
                        if only_nonzeros and abs(tensor[i, j, k, l, m]) < tol:
                            continue
                        if is_cmplx:
                            fmt_string = (fmt_int + space) * 5 + \
                                (fmt_float + space) * 2 + '\n'
                            f.write(fmt_string.format(i + 1, j + 1, k + 1, l + 1, m + 1,
                                                      tensor[i, j, k, l, m].real, tensor[i, j, k, l, m].imag))
                        else:
                            fmt_string = (fmt_int + space) * 4 + \
                                fmt_float + space + '\n'
                            f.write(fmt_string.format(i + 1, j + 1, k +
                                                      1, l + 1, m + 1, tensor[i, j, k, l, m]))
    f.close()


def write_tensor(tensor, fname, only_nonzeros=True, tol=1E-10, fmt_int='{:10d}',
                 fmt_float='{:.15f}'):
    """
    Write :math:`n` -dimension numpy array to file, currently, :math:`n` can be 1, 2, 3, 4, 5.

    Parameters
    ----------
    tensor : :math:`n` d float or complex array
        The array needs to be written.

    fname : str
        File name.

    only_nonzeros : logical (default: True)
        Only write nonzero elements.

    tol : float (default: 1E-10)
        Only write the elements when their absolute value are larger than tol
        and only_nonzeros=True.

    fmt_int : str (default: '{:10d}')
        The format for printing integer numbers.

    fmt_float : str (default: '{:.15f}')
        The format for printing float numbers.

    """

    ndim = tensor.ndim
    if ndim == 1:
        write_tensor_1(tensor, fname, only_nonzeros=only_nonzeros, tol=tol,
                       fmt_int=fmt_int, fmt_float=fmt_float)
    elif ndim == 2:
        write_tensor_2(tensor, fname, only_nonzeros=only_nonzeros, tol=tol,
                       fmt_int=fmt_int, fmt_float=fmt_float)
    elif ndim == 3:
        write_tensor_3(tensor, fname, only_nonzeros=only_nonzeros, tol=tol,
                       fmt_int=fmt_int, fmt_float=fmt_float)
    elif ndim == 4:
        write_tensor_4(tensor, fname, only_nonzeros=only_nonzeros, tol=tol,
                       fmt_int=fmt_int, fmt_float=fmt_float)
    elif ndim == 5:
        write_tensor_5(tensor, fname, only_nonzeros=only_nonzeros, tol=tol,
                       fmt_int=fmt_int, fmt_float=fmt_float)
    else:
        print("error in write_tensor: ndim >5, not implemented !")
        sys.exit()


def write_emat(emat, fname, tol=1E-12, fmt_int='{:10d}', fmt_float='{:.15f}'):
    """
    Write the nonzeros of the rank-2 hopping matrices to file.
    The first line is the number of nonzeros, and the following lines are the nonzero elements.
    This file will be read by ed.x, xas.x or rixs.x.

    Parameters
    ----------
    emat : 2d complex array
        The array to be written.

    fname : str
        File name.

    tol : float
        Precision.

    fmt_int : str (default: '{:10d}')
        Format for printing integer numbers.

    fmt_float : str (default: '{:.15f}')
        Format for printing float numbers.
    """

    a1, a2 = np.nonzero(abs(emat) > tol)
    nonzero = np.stack((a1, a2), axis=-1)

    space = "  "
    fmt_string = (fmt_int + space) * 2 + (fmt_float + space) * 2 + '\n'
    f = open(fname, 'w')
    if len(nonzero) == 0:
        f.write("{:10d}\n".format(1))
        f.write(fmt_string.format(1, 1, 0.0, 0.0))
    else:
        f.write("{:20d}\n".format(len(nonzero)))
        for i, j in nonzero:
            f.write(fmt_string.format(i + 1, j + 1,
                                      emat[i, j].real, emat[i, j].imag))
    f.close()


def write_umat(umat, fname, tol=1E-12, fmt_int='{:10d}', fmt_float='{:.15f}'):
    """
    Write the nonzeros of the rank-4 Coulomb U tensor to file.
    The first line is the number of nonzeros, and the following lines are the nonzero elements.
    This file will be read by ed.x, xas.x or rixs.x.

    Parameters
    ----------
    umat : 4d complex array
        The array to be written.

    fname : str
        File name.

    tol : float (default: 1E-12)
        Precision.

    fmt_int : str (default: '{:10d}')
        Format for printing integer numbers.

    fmt_float : str (default: '{:.15f}')
        Format for printing float numbers.
    """

    a1, a2, a3, a4 = np.nonzero(abs(umat) > tol)
    nonzero = np.stack((a1, a2, a3, a4), axis=-1)

    space = "  "
    fmt_string = (fmt_int + space) * 4 + (fmt_float + space) * 2 + '\n'
    f = open(fname, 'w')
    if len(nonzero) == 0:
        f.write("{:10d}\n".format(1))
        f.write(fmt_string.format(1, 1, 1, 1, 0.0, 0.0))
    else:
        f.write("{:20d}\n".format(len(nonzero)))
        for i, j, k, l in nonzero:
            f.write(fmt_string.format(i + 1, j + 1, k + 1, l + 1,
                                      umat[i, j, k, l].real, umat[i, j, k, l].imag))
    f.close()
