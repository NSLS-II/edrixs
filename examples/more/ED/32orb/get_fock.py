#!/usr/bin/env python

import sys
from edrixs.fock_basis import  get_fock_basis_by_NSz

if __name__ == "__main__":
    Sz_list=[1, -1]*16
    basis=get_fock_basis_by_NSz(32, 16, Sz_list)
    tot_sz=int(sys.argv[1])
    print("Total Sz: ", tot_sz)
    for key, val in list(basis.items()):
        if key==tot_sz:
            if len(val) > 0: 
                val.sort()
                fname="fock_i.in"
                f=open(fname, 'w')
                print(len(val), file=f)
                for i in val:
                    print(i, file=f)
                f.close()
            else:
                print("ERROR: Wrong total Sz, check the argument and try again !")
                sys.exit()
            break
