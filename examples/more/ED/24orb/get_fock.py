#!/usr/bin/env python

import sys
from fock_basis import  write_fock_dec_by_N

if __name__ == "__main__":
    write_fock_dec_by_N(24, 12, "fock_i.in")
