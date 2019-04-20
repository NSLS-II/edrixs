#!/usr/bin/env python

import edrixs
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

edrixs.ed_fsolver(rank, size)

