#!/usr/bin/env python

import copy
import numpy as np
from scipy              import   signal
from photon_transition  import   dipole_polvec_rixs
from utils              import   boltz_dist
from rixs_utils         import   scattering_mat

if __name__ == "__main__":
    '''
    Purpose: This example shows how to calculate RIXS spectrum.
             This example use purely python code.
    '''

    # PARAMETERS
    #-----------
    # the parameters for the experimental RIXS geometry are taken from [PRL 117, 147401 (2016)]
    # the incident angle of X-ray
    thin, thout = 15/180.0*np.pi,  75/180.0*np.pi
    # azimuthal angle
    phi = 0.0
    # core-hole life-time broadening
    gamma_n = 0.20
    # resolution of RIXS excitations
    gamma_f = 0.10
    # energy offset of the incident X-ray
    om_offset = 857.4
    # set energy mesh of the incident X-ray (eV)
    # L3 edge
    om1, om2 = -5.9,-0.9
    # L2 dege
    #om1, om2 = 10.9, 14.9
    nom = 100
    om_mesh = np.linspace(om1, om2, nom)
    # energy loss mesh
    neloss = 1000
    eloss_mesh = np.linspace(-0.5, 5.0, neloss)
    # ground state list
    gs_list=list(range(0, 3))
    # temperature
    T = 300 
    # END of PARAMETERS
    #------------------

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

    # We calculate RIXS for \pi-\pi, \pi-\sigma, \sigma-\pi, \sigma-\sigma polarizations
    rixs = np.zeros((4, neloss, nom), dtype=np.float64)
    gs_prob = boltz_dist([eval_i[i] for i in gs_list], T) 
    polvec_list = [(0,0), (0,np.pi/2.0), (np.pi/2.0, 0), (np.pi/2.0, np.pi/2.0)]
    
    print("edrixs >>> calculating RIXS ...")
    for i, om_inc in enumerate(om_mesh):
        print("    incident X-ray energy:  ", i, "   ", om_inc)
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
    # gaussian broadening
    inc_res = 0.17
    emi_res = 0.12 
    gauss = np.zeros((neloss, nom))
    mid_in = ( min(om_mesh) + max(om_mesh) ) /2.0
    mid_out  = ( min(eloss_mesh)+max(eloss_mesh)) /2.0
    for i in range(nom):
        for j in range(neloss):
            gauss[j,i] = 1/(2.0*np.pi*inc_res*emi_res)*np.exp(-((mid_in-om_mesh[i])**2/(2*inc_res**2) + (mid_out-eloss_mesh[j])**2/(2*emi_res**2)))
    for i in range(4):
        rixs[i,:,:] = signal.fftconvolve(rixs[i,:,:], gauss, mode = 'same')
    print("edrixs >>> done !")

    f=open('rixs.dat', 'w')
    for i in range(neloss):
        for j in range(nom):
            str_form = "{:20.10f}"*6 +"\n"
            line=str_form.format(eloss_mesh[i], om_mesh[j]+om_offset, rixs[0,i,j], rixs[1,i,j], rixs[2,i,j], rixs[3,i,j])
            f.write(line)
    f.close()
