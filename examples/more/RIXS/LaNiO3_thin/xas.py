#!/usr/bin/env python

import numpy as np

from edrixs.photon_transition import dipole_polvec_xas
from edrixs.utils import boltz_dist

if __name__ == "__main__":
    '''
    Purpose: This example shows how to calculate XAS spectrum.
             XAS(om) = \sum_{i,n} |<n|D|i>|^2 * Gamma/Pi / [ (om-(E_n-E_i))^2 + Gamma^2 ] * Prob(i)
             This example use purely python code.
    '''
    # PARAMETERS
    #-----------
    # the incident angle of X-ray
    thin = 15/180.0*np.pi
    # azimuthal angle
    phi = 0.0
    # core-hole life-time broadening
    gamma = 0.2
    # energy mesh of the incident X-ray (eV)
    om1, om2, nom = -10, 20, 1000
    om_mesh = np.linspace(om1, om2, nom)
    # In our calculations, we set the energy level of 2p core to zero, so we need to
    # adjust om_offset to match the experimental X-ray energy, it can be roughly estimated by, for example, Ni-L2/L3 edge,
    # om_offset = (2*E_L3 + E_L2)/3, E_L3 and E_L2 can be found in software, such as, Hephaestus [https://bruceravel.github.io/demeter/aug/hephaestus.html].
    # for Ni, E_L3=852.7 eV and E_L2=870 eV, so the calculated om_offset is 858.47 eV, however, due to the different 
    # crystal environment, we fine tune it to be 857.40 according to the experimental XAS and RIXS data.
    om_offset = 857.40
    # which initial states are used to calculate the spectra
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
    data  = np.loadtxt('trans_mat.dat')
    trans_mat = data[:,3].reshape((3, ncfgs_n, ncfgs_i)) + 1j * data[:,4].reshape((3, ncfgs_n, ncfgs_i))

    # We calculate XAS for 5 different polarizations
    xas = np.zeros((5, nom), dtype=np.float64)
    # get boltzmann distribution at T for these ground states 
    gs_prob = boltz_dist([eval_i[i] for i in gs_list], T) 

    polvec =[]
    # (1) linear \pi-pol, the polarization vector is in the scattering plane
    pol_pi    = dipole_polvec_xas(thin, phi, 0)
    polvec.append(pol_pi)
    # (2) linear \sigma-pol, the polarization vector is normal to the scattering plane
    pol_sigma = dipole_polvec_xas(thin, phi, np.pi/2.0)
    polvec.append(pol_sigma)

    # (3) left circular polarization
    polvec.append( (pol_pi + 1j * pol_sigma) / np.sqrt(2.0) ) 
    # (4) right circular polarization
    polvec.append( (pol_pi - 1j * pol_sigma) / np.sqrt(2.0) ) 
    # (5) isotropic 
    polvec.append( np.array([1,1,1])/np.sqrt(3.0) )

    print("edrixs >>> calculating XAS ...")
    for m, ei in enumerate(polvec):
        for i, om_inc in enumerate(om_mesh):
            for j in gs_list:
                scatt_mat = trans_mat[0,:,j] * ei[0] + trans_mat[1,:,j] * ei[1] + trans_mat[2,:,j] * ei[2]        
                xas[m, i] += np.sum( np.abs( scatt_mat )**2  * gamma/np.pi / ((om_inc - (eval_n[:] - eval_i[j]))**2 + gamma**2) ) * gs_prob[j]
    print("edrixs >>> Done !")

    f=open('xas.dat', 'w')
    f.write("#       omega_inc              pi-pol             sigma-pol           left-pol            right-pol          isotropic \n")
    for i in range(nom):
        f.write("{:20.10f}{:20.10f}{:20.10f}{:20.10f}{:20.10f}{:20.10f}\n".format(om_offset+om_mesh[i], xas[0,i], xas[1,i], xas[2,i], xas[3,i], xas[4,i]))
    f.close() 

