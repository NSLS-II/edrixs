#!/usr/bin/env python

import shutil
import argparse
import edrixs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get fock basis by total occupancy number")
    parser.add_argument('-norb', type=int, default=1, help='Total number of valence orbitals')
    parser.add_argument('-noccu', type=int, default=1, help='Total occupancy number')
    args = parser.parse_args()
    norbs, occu = args.norb, args.noccu

    edrixs.write_fock_dec_by_N(norbs, occu, 'ed/fock_i.in')
    shutil.copy('ed/fock_i.in', 'xas/fock_i.in')
    shutil.copy('ed/fock_i.in', 'rixs_pp/fock_i.in')
    shutil.copy('ed/fock_i.in', 'rixs_ps/fock_i.in')
    shutil.copy('ed/fock_i.in', 'rixs_pp/fock_f.in')
    shutil.copy('ed/fock_i.in', 'rixs_ps/fock_f.in')

    edrixs.write_fock_dec_by_N(norbs, occu + 1, 'xas/fock_n.in')
    shutil.copy('xas/fock_n.in', 'rixs_pp/fock_n.in')
    shutil.copy('xas/fock_n.in', 'rixs_ps/fock_n.in')
