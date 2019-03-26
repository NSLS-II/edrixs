#!/usr/bin/env python

import numpy as np
from photon_transition import dipole_polvec_xas
from utils import boltz_dist

if __name__ == "__main__":
    '''
    Purpose: This example shows how to calculate XAS spectrum.
             XAS(om) = \sum_{i,n} |<n|D|i>|^2 * Gamma/Pi / [ (om-(E_n-E_i))^2 + Gamma^2 ] * Prob(i)
             This example use purely python code.
    '''
    # PARAMETERS
    #-----------
    # core-hole life-time broadening
    gamma = 2.5
    # energy mesh of the incident X-ray (eV)
    om1, om2, nom = -600, 1100, 10000
    #om1, om2, nom = -520, -480, 10000
    om_mesh = np.linspace(om1, om2, nom)
    # energy shift of incident X-ray
    om_shift = 11380
    # which initial states are used 
    gs_list=list(range(0, 4))
    # temperature
    T = 300 
    # isotropic polarization
    ei = np.array([1,1,1])/np.sqrt(3.0)

    # END of PARAMETERS
    #------------------

    # load data, the eigenvalues of the initial and the intermediate Hamiltonian, and the transition matrix 
    data = np.loadtxt('eval_i.dat') 
    eval_i = data[:,1]
    data = np.loadtxt('eval_n.dat') 
    eval_n = data[:,1]
    ncfgs_n, ncfgs_i = len(eval_n), len(eval_i)
    data  = np.loadtxt('trans_mat.dat')
    trans_mat = data[:,3].reshape((3, ncfgs_n, ncfgs_i)) + 1j * data[:,4].reshape((3, ncfgs_n, ncfgs_i))

    print("edrixs >>> calculating XAS ...")
    xas = np.zeros(nom, dtype=np.float64)
    # get boltzmann distribution at T for these ground states 
    gs_prob = boltz_dist([eval_i[i] for i in gs_list], T) 

    for i, om_inc in enumerate(om_mesh):
        for j in gs_list:
            scatt_mat = trans_mat[0,:,j] * ei[0] + trans_mat[1,:,j] * ei[1] + trans_mat[2,:,j] * ei[2]        
            xas[i] += np.sum( np.abs( scatt_mat )**2  * gamma/np.pi / ((om_inc - (eval_n[:] - eval_i[j]))**2 + gamma**2) ) * gs_prob[j]

    f=open('xas.dat', 'w')
    for i in range(nom):
        f.write("{:20.10f}{:20.10f}\n".format(om_shift+om_mesh[i], xas[i]))
    f.close() 

    print("edrixs >>> Done !")

