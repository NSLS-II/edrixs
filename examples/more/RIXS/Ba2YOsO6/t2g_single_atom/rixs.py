#!/usr/bin/env python

import copy
import numpy as np
from scipy import signal

from edrixs.photon_transition import dipole_polvec_rixs
from edrixs.utils             import boltz_dist
from edrixs.rixs_utils        import scattering_mat

if __name__ == "__main__":
    '''
    Purpose: This example shows how to calculate RIXS spectrum.
             This example use purely python code.
    '''
    # PARAMETERS
    # -----------
    # the incident angle of X-ray
    thin, thout = 45/180.0*np.pi,  45/180.0*np.pi
    # azimuthal angle
    phi = 0.0
    # core-hole life-time broadening
    gamma_n = 2.5
    # resolution of RIXS excitations
    gamma_f = 0.075
    
    # energy shift of the incident X-ray
    om_shift = 11380

    # energy mesh of the incident X-ray (eV)
    # L3 edge
    om1, om2 = -508,-500
    # L2 dege
    #om1, om2 = 1000, 1015

    nom = 100
    om_mesh = np.linspace(om1, om2, nom)

    # energy loss mesh
    eloss_1, eloss_2 = -0.5, 2.0
    neloss = 1000
    eloss_mesh = np.linspace(eloss_1, eloss_2, neloss)

    # ground state list
    gs_list=list(range(0, 4))
    # temperature
    T = 300 

    # END of PARAMETERS
    # -----------------


    # load data, the eigenvalues of the initial and the intermediate Hamiltonian, and the transition matrix 
    data = np.loadtxt('eval_i.dat') 
    eval_i = data[:,1]
    data = np.loadtxt('eval_n.dat') 
    eval_n = data[:,1]
    ncfgs_n, ncfgs_i = len(eval_n), len(eval_i)

    # the transition operator for the absorption process
    data  = np.loadtxt('trans_mat.dat')
    trans_mat_abs = data[:,3].reshape((3, ncfgs_n, ncfgs_i)) + 1j * data[:,4].reshape((3, ncfgs_n, ncfgs_i))
    # the transition operator for the emission process
    trans_mat_emi = np.zeros((3, ncfgs_i, ncfgs_n), dtype=np.complex128)
    for i in range(3):
        trans_mat_emi[i] = np.conj(np.transpose(trans_mat_abs[i]))


    print("edrixs >>> calculating RIXS ...")
    rixs = np.zeros((4, neloss, nom), dtype=np.float64)
    gs_prob = boltz_dist([eval_i[i] for i in gs_list], T) 

    # We calculate RIXS for \pi-\pi, \pi-\sigma, \sigma-\pi, \sigma-\sigma polarizations
    polvec_list = [(0,0), (0,np.pi/2.0), (np.pi/2.0, 0), (np.pi/2.0, np.pi/2.0)]
    
    for i, om_inc in enumerate(om_mesh):
        print(i, om_inc)
        F_fi = scattering_mat(eval_i, eval_n, trans_mat_abs[:,:,gs_list], trans_mat_emi, om_inc, gamma_n)
        for j, (alpha, beta) in enumerate(polvec_list):  
            ei, ef = dipole_polvec_rixs(thin, thout, phi, alpha, beta)
            F_magnitude = np.zeros((ncfgs_i, len(gs_list)), dtype=np.complex128)
            for m in range(3):
                for n in range(3):
                    F_magnitude[:,:] += ef[m] * F_fi[m,n] * ei[n]
            for m in gs_list:
                for n in range(ncfgs_i):
                    rixs[j, :, i] += np.abs(F_magnitude[n,m])**2 * gamma_f/np.pi / ( (eloss_mesh-(eval_i[n]-eval_i[m]))**2 + gamma_f**2 ) * gs_prob[m] 

    f=open('rixs.dat', 'w')
    for i in range(neloss):
        for j in range(nom):
            str_form = "{:20.10f}"*6 +"\n"
            line=str_form.format(eloss_mesh[i], om_mesh[j]+om_shift, rixs[0,i,j], rixs[1,i,j], rixs[2,i,j], rixs[3,i,j])
            f.write(line)
    f.close()
    print("edrixs >>> done !")
