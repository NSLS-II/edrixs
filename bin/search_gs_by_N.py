#!/usr/bin/env python

import subprocess
import numpy as np
import argparse
import edrixs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search ground states by total occupancy number for Anderson impurity model.")
    parser.add_argument('-ntot', type=int, default=1, help='Total number of valence orbitals')
    parser.add_argument('-nimp', type=int, default=1, help='Number of impurity valence orbitals')
    parser.add_argument('-N1', type=int, default=0, help='Occupancy number')
    parser.add_argument('-N2', type=int, default=0, help='Occupancy number')
    parser.add_argument('-maxiter', type=int, default=1000,
                        help='Maximum iterations of Lanczos process')
    parser.add_argument('-tol', type=float, default=1E-10,
                        help='Tolerance for eigenvalue of ground state')
    parser.add_argument('-mpi_cmd', type=str, default='mpi_cmd.dat',
                        help='File containing the MPI command')
    args = parser.parse_args()

    occu_start, occu_end = min(args.N1, args.N2), max(args.N1, args.N2)
    Norb = args.ntot
    Nimp = args.nimp
    maxiter = args.maxiter
    tol = args.tol
    ied = 1
    nev = 1
    nvector = 1

    config = [
        "&control",
        "num_val_orbs=" + str(Norb),
        "ed_solver=" + str(ied),
        "neval=" + str(nev),
        "nvector=" + str(nvector),
        "maxiter=" + str(maxiter),
        "eigval_tol=" + str(tol),
        "idump=.false.",
        "&end"
    ]
    f = open('config.in', 'w')
    for item in config:
        f.write(item + "\n")
    f.close()

    flog = open("search_gs.log", 'w')
    res = []
    for occu in range(occu_start, occu_end + 1):
        ndim = edrixs.write_fock_dec_by_N(Norb, occu, 'fock_i.in')
        f = open(args.mpi_cmd, 'r')
        mpi_args = f.readline().split()
        f.close()
        # For mpi_cmd: mpirun -np number_of_cpus  ed.x
        # if number_fo_cpus > ndim, please reduce the number_of_cpus to ndim
        subprocess.check_call(mpi_args)

        eig_f = open('eigvals.dat', 'r')
        e_gs = float(eig_f.readline().split()[1])
        eig_f.close()

        data = np.loadtxt('denmat.dat')
        den = (data[:, 3].reshape((nvector, Norb, Norb)) +
               1j * data[:, 4].reshape((nvector, Norb, Norb)))
        nd = np.sum(den[0].diagonal()[0:Nimp])
        nd = nd.real
        print(occu, ndim, e_gs, nd, file=flog)
        flog.flush()
        res.append((occu, ndim, e_gs, nd))

    res.sort(key=lambda x: x[2])
    f = open('result.dat', 'w')
    for item in res:
        f.write("{:10d}{:10d}{:20.10f}{:20.10f}\n".format(item[0], item[1], item[2], item[3]))
    f.close()
