#!/usr/bin/env python


import sys
import numpy as np
import edrixs
from edrixs.fedrixs import ed_fsolver
from mpi4py import MPI


def get_hopping_coulomb():
    N_site = 16
    norbs = 32
    U, t = 4.0, -1.0

    umat = np.zeros((norbs, norbs, norbs, norbs), dtype=np.complex128)
    for i in range(N_site):
        off = i * 2
        umat[off, off + 1, off + 1, off] = U

    hopp = np.zeros((N_site, N_site), dtype=np.complex128)
    indx = [
        [3, 1, 12, 4], [0, 2, 13, 5], [1, 3, 14, 6], [2, 0, 15, 7],
        [7, 5, 0, 8], [4, 6, 1, 9], [5, 7, 2, 10], [6, 4, 3, 11],
        [11, 9, 4, 12], [8, 10, 5, 13], [9, 11, 6, 14], [10, 8, 7, 15],
        [15, 13, 8, 0], [12, 14, 9, 1, ], [13, 15, 10, 2], [14, 12, 11, 3]
    ]

    for i, item in enumerate(indx):
        hopp[i, item[0]] = hopp[i, item[1]] = hopp[i, item[2]] = hopp[i, item[3]] = t
    hopping = np.zeros((norbs, norbs), dtype=np.complex128)
    hopping[0:norbs:2, 0:norbs:2] = hopp
    hopping[1:norbs:2, 1:norbs:2] = hopp

    edrixs.write_emat(hopping, "hopping_i.in", 1E-10)
    edrixs.write_umat(umat, "coulomb_i.in", 1E-10)


def get_config():
    config_in = [
        "&control",
        "ed_solver    = 1",
        "num_val_orbs = 32",
        "neval        = 1",
        "nvector      = 1",
        "maxiter      = 500",
        "eigval_tol   = 1E-10",
        "idump        = .false.",
        "&end"
    ]

    f = open('config.in', 'w')
    for line in config_in:
        f.write(line + "\n")
    f.close()


def get_fock(tot_sz):
    Sz_list = [1, -1] * 16
    basis = edrixs.get_fock_basis_by_NSz(32, 16, Sz_list)
    print("Total Sz: ", tot_sz)
    for key, val in list(basis.items()):
        if key == tot_sz:
            if len(val) > 0:
                val.sort()
                fname = "fock_i.in"
                f = open(fname, 'w')
                print(len(val), file=f)
                for i in val:
                    print(i, file=f)
                f.close()
            else:
                print("ERROR: Wrong total Sz, check the argument and try again !")
                sys.exit()
            break


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    fcomm = comm.py2f()

    if rank == 0:
        get_hopping_coulomb()
        get_config()
        tot_sz = int(sys.argv[1])
        get_fock(tot_sz)

    comm.Barrier()
    print("edrixs >>> Running ED ...")
    ed_fsolver(fcomm, rank, size)
