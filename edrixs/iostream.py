#!/usr/bin/env python
import numpy as np

def write_nonzeros_tensor1(tensor, fname, tol=1E-6):
    (n1, ) = tensor.shape
    is_cmplx = False
    if tensor.dtype==np.complex or tensor.dtype==np.complex128:
        is_cmplx = True    

    f=open(fname, 'w')
    for i in range(n1):
        if abs(tensor[i]) > tol:
            if is_cmplx:
                f.write("{:10d}{:20.12f}{:20.12f}\n".format(i+1, tensor[i].real, tensor[i].imag))
            else:
                f.write("{:10d}{:20.12f}\n".format(i+1, tensor[i]))
    f.close()

def write_nonzeros_tensor2(tensor, fname, tol=1E-6):
    (n1, n2) = tensor.shape
    is_cmplx = False
    if tensor.dtype==np.complex or tensor.dtype==np.complex128:
        is_cmplx = True    
    f=open(fname, 'w')
    for i in range(n1):
        for j in range(n2):
            if abs(tensor[i,j]) > tol:
                if is_cmplx:
                    f.write("{:10d}{:10d}{:20.12f}{:20.12f}\n".format(i+1, j+1, tensor[i,j].real, tensor[i,j].imag))
                else:
                    f.write("{:10d}{:10d}{:20.12f}\n".format(i+1, j+1, tensor[i,j]))
    f.close()

def write_nonzeros_tensor3(tensor, fname, tol=1E-6):
    (n1, n2, n3) = tensor.shape
    is_cmplx = False
    if tensor.dtype==np.complex or tensor.dtype==np.complex128:
        is_cmplx = True    
    f=open(fname, 'w')
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                if abs(tensor[i,j,k]) > tol:
                    if is_cmplx:
                        f.write("{:10d}{:10d}{:10d}{:20.12f}{:20.12f}\n".format(i+1, j+1, k+1, tensor[i,j,k].real, tensor[i,j,k].imag))
                    else:
                        f.write("{:10d}{:10d}{:10d}{:20.12f}\n".format(i+1, j+1, k+1, tensor[i,j,k]))
    f.close()

def write_nonzeros_tensor4(tensor, fname, tol=1E-6):
    (n1, n2, n3, n4) = tensor.shape
    is_cmplx = False
    if tensor.dtype==np.complex or tensor.dtype==np.complex128:
        is_cmplx = True    
    f=open(fname, 'w')
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                for l in range(n4):
                    if abs(tensor[i,j,k,l]) > tol:
                        if is_cmplx:
                            f.write("{:10d}{:10d}{:10d}{:10d}{:20.12f}{:20.12f}\n".format(i+1, j+1, k+1, l+1, tensor[i,j,k,l].real, tensor[i,j,k,l].imag))
                        else:
                            f.write("{:10d}{:10d}{:10d}{:10d}{:20.12f}\n".format(i+1, j+1, k+1, l+1, tensor[i,j,k,l]))
    f.close()


def write_tensor1(tensor, fname):
    (n1,) = tensor.shape
    is_cmplx = False
    if tensor.dtype==np.complex or tensor.dtype==np.complex128:
        is_cmplx = True
    f=open(fname, 'w')
    for i in range(n1):
        if is_cmplx:
            f.write("{:10d}{:20.12f}{:20.12f}\n".format(i+1, tensor[i].real, tensor[i].imag))
        else:
            f.write("{:10d}{:20.12f}\n".format(i+1, tensor[i]))
    f.close()

def write_tensor2(tensor, fname):
    n1,n2 = tensor.shape
    is_cmplx = False
    if tensor.dtype==np.complex or tensor.dtype==np.complex128:
        is_cmplx = True
    f=open(fname, 'w')
    for i in range(n1):
        for j in range(n2):
            if is_cmplx:
                f.write("{:10d}{:10d}{:20.12f}{:20.12f}\n".format(i+1, j+1, tensor[i,j].real, tensor[i,j].imag))
            else:
                f.write("{:10d}{:10d}{:20.12f}\n".format(i+1, j+1, tensor[i,j]))
    f.close()

def write_tensor3(tensor, fname):
    n1,n2,n3 = tensor.shape
    is_cmplx = False
    if tensor.dtype==np.complex or tensor.dtype==np.complex128:
        is_cmplx = True
    f=open(fname, 'w')
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                if is_cmplx:
                    f.write("{:10d}{:10d}{:10d}{:20.12f}{:20.12f}\n".format(i+1, j+1, k+1, tensor[i,j,k].real, tensor[i,j,k].imag))
                else:
                    f.write("{:10d}{:10d}{:10d}{:20.12f}\n".format(i+1, j+1, k+1, tensor[i,j,k]))
    f.close()

def write_tensor4(tensor, fname):
    n1,n2,n3,n4 = tensor.shape
    is_cmplx = False
    if tensor.dtype==np.complex or tensor.dtype==np.complex128:
        is_cmplx = True
    f=open(fname, 'w')
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                for l in range(n4):
                    if is_cmplx:
                        f.write("{:10d}{:10d}{:10d}{:10d}{:20.12f}{:20.12f}\n".format(i+1, j+1, k+1, l+1, tensor[i,j,k,l].real, tensor[i,j,k,l].imag))
                    else:
                        f.write("{:10d}{:10d}{:10d}{:10d}{:20.12f}\n".format(i+1, j+1, k+1, l+1, tensor[i,j,k,l]))
    f.close()

def write_tensor(tensor, fname):
    """
    Write :math:`n` -dimension numpy array to file, currently, :math:`n` can be 1, 2, 3, 4.

    Parameters
    ----------
    tensor : :math:`n` d float or complex array
        The array needs to be written.

    fname : str
        File name.
    """

    ndim = tensor.ndim
    if ndim == 1:
        write_tensor1(tensor, fname)
    elif ndim == 2:
        write_tensor2(tensor, fname)
    elif ndim == 3:
        write_tensor3(tensor, fname)
    elif ndim == 4:
        write_tensor4(tensor, fname)
    else:
        print("NOT implemented!")
 
def write_tensor_nonzeros(tensor, fname, tol=1E-6):
    """
    Write nonzeros of the elements of a :math:`n` -dimension numpy array to file, currently, :math:`n` can be 1, 2, 3, 4.

    Parameters
    ----------
    tensor : :math:`n` d float or complex array
        The array needs to be written.

    fname : str
        File name.

    tol : float
        The precision, the elements whose absolute value is large than tol will be written.
    """

    ndim = tensor.ndim
    if ndim == 1:
        write_nonzeros_tensor1(tensor, fname, tol)
    elif ndim == 2:
        write_nonzeros_tensor2(tensor, fname, tol)
    elif ndim == 3:
        write_nonzeros_tensor3(tensor, fname, tol)
    elif ndim == 4:
        write_nonzeros_tensor4(tensor, fname, tol)
    else:
        print("NOT implemented!")

def write_emat(emat, fname, tol=1E-12):
    """
    Write the nonzeros of a matrix to file. The first line is the number of nonzeros, and the following lines are the nonzero elements.
    This file will be read by ed.x, xas.x or rixs.x.

    Parameters
    ----------
    emat : 2d complex array
        The array to be written.

    fname : str
        File name.

    tol : float
        Precision.
    """

    a1,a2 = np.nonzero(abs(emat) > tol)
    nonzero = np.stack((a1,a2), axis=-1)

    f=open(fname, 'w')
    if len(nonzero) == 0:
        f.write("{:10d}\n".format(1))
        f.write("{:10d}{:10d}{:20.12f}{:20.12f}\n".format(1, 1, 0.0, 0.0))
    else:
        f.write("{:20d}\n".format(len(nonzero)))
        for i, j in nonzero:
            f.write("{:10d}{:10d}{:20.12f}{:20.12f}\n".format(i+1, j+1, emat[i,j].real, emat[i,j].imag))
    f.close()

def write_umat(umat, fname, tol=1E-12):
    """
    Write the nonzeros of a rank-4 tensor to file. The first line is the number of nonzeros, and the following lines are the nonzero elements.
    This file will be read by ed.x, xas.x or rixs.x.

    Parameters
    ----------
    umat : 4d complex array
        The array to be written.

    fname : str
        File name.

    tol : float
        Precision.
    """

    a1,a2,a3,a4 = np.nonzero(abs(umat) > tol)
    nonzero = np.stack((a1,a2,a3,a4), axis=-1)

    f=open(fname, 'w')
    if len(nonzero) == 0:
        f.write("{:10d}\n".format(1))
        f.write("{:10d}{:10d}{:10d}{:10d}{:20.12f}{:20.12f}\n".format(1, 1, 1, 1, 0.0, 0.0))
    else:
        f.write("{:20d}\n".format(len(nonzero)))
        for i, j, k, l in nonzero:
            f.write("{:10d}{:10d}{:10d}{:10d}{:20.12f}{:20.12f}\n".format(i+1, j+1, k+1, l+1, umat[i,j,k,l].real, umat[i,j,k,l].imag))
    f.close()
