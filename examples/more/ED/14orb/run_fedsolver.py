#!/usr/bin/env python

from edrixs.fedrixs import ed_fsolver
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
fcomm = comm.py2f()
ed_fsolver(fcomm, rank, size)
