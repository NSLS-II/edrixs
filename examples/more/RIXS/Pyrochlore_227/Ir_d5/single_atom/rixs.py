#!/usr/bin/env python
import sys
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from edrixs.utils             import boltz_dist
from edrixs.rixs_utils        import scattering_mat
from edrixs.photon_transition import get_wavevector_rixs,dipole_polvec_rixs

np.set_printoptions(precision=10, linewidth=200, suppress=True)

if __name__ == '__main__':
    a=10.2
    atom_pos = a * np.array([[0.50, 0.50, 0.50], 
                             [0.25, 0.25, 0.50],
                             [0.25, 0.50, 0.25],
                             [0.50, 0.25, 0.25]], dtype=np.float64) 
 
    ncfgs_gs, ncfgs_ex = 6,6
    data=np.loadtxt('eval_i.dat')
    eval_gs = data[:,1]
    data=np.loadtxt('eval_n.dat')
    eval_ex = data[:,1]

    data=np.loadtxt('trans_mat.dat')
    trans_mat_abs = data[:,4].reshape((4, 3, ncfgs_ex, ncfgs_gs)) + 1j * data[:,5].reshape((4,3, ncfgs_ex, ncfgs_gs))
    trans_mat_emi = np.zeros((4, 3, ncfgs_gs, ncfgs_ex),dtype=np.complex128)
    for i in range(4):
        for j in range(3):
            trans_mat_emi[i,j] = np.conj(np.transpose(trans_mat_abs[i,j]))
    gamma_c, gamma = 5.0/2.0, 0.08
    phi, thin, thout = 0.0,  45/180.0*np.pi, 45/180.0*np.pi
    pol_vec=[(0,0), (0,np.pi/2.0)]
    loc_scatt  = np.transpose( np.array([[ 1/np.sqrt(6.0), 1/np.sqrt(6.0), -2/np.sqrt(6.0)], 
                                         [-1/np.sqrt(2.0), 1/np.sqrt(2.0),               0],
                                         [ 1/np.sqrt(3.0), 1/np.sqrt(3.0),  1/np.sqrt(3.0)]]) )
    omi_shift = 11744.0
    omi_mesh = np.linspace(-533, -525, 100)
    eloss_mesh = np.linspace(-0.1, 1.2,1000)
    gs_list = range(0,2)
    gs_dist =  boltz_dist([eval_gs[i] for i in gs_list], 30)

    rixs=np.zeros((len(pol_vec),len(eloss_mesh), len(omi_mesh)), dtype=np.float64)
    for omi_ind, omi in enumerate(omi_mesh):
        print(omi_ind, omi, omi+omi_shift)
        Ffg_atom = np.zeros((4,3,3, ncfgs_gs, len(gs_list)), dtype=np.complex128)
        for i in range(0,4):
            abs_trans = copy.deepcopy(trans_mat_abs[i])
            emi_trans = copy.deepcopy(trans_mat_emi[i])
            Ffg_atom[i]=scattering_mat(eval_gs, eval_ex, abs_trans[:,:,gs_list], emi_trans, omi, gamma_c)
        print('------------------------')
        for ipol, (alpha,beta) in enumerate(pol_vec):
            ein, eout = omi+omi_shift, omi+omi_shift
            ei, ef =  dipole_polvec_rixs(thin, thout, phi, alpha, beta, loc_scatt)
            kin, kout = get_wavevector_rixs(thin, thout, phi, ein, eout, loc_scatt)
            Q = kout - kin
            phase = np.zeros(4, dtype=np.complex128)
            for i in range(0,4):
                phase[i] = np.exp(-1j * np.dot(Q, atom_pos[i]))
            Ffg=np.zeros((ncfgs_gs, len(gs_list)), dtype=np.complex128)
            for i in range(0,4):
                for j in range(3):
                    for k in range(3):
                        Ffg[:,:] += ef[j] * Ffg_atom[i,j,k]*phase[i] * ei[k]
            for i in gs_list:
                for j in range(ncfgs_gs):
                    rixs[ipol, :, omi_ind] += np.abs(Ffg[j,i])**2 * gamma/(2*np.pi) / ( (eloss_mesh-(eval_gs[j]-eval_gs[i]))**2 + (gamma/2.0)**2 )*gs_dist[i]
    # save rixs cut
    f=open('rixs_tot.dat', 'w')
    rixs_tot = np.zeros(len(eloss_mesh), dtype=np.float64)
    for i in range(len(eloss_mesh)):
        rixs_tot[i] = np.sum(rixs[0,i,:] + rixs[1,i,:]) / len(omi_mesh)
        line="{:20.10f}{:20.10f}\n".format(eloss_mesh[i], rixs_tot[i]) 
        f.write(line)
    f.close()

    # plot rixs map
    font = {'family' : 'Times New Roman' ,
    	    'weight' : 'normal' ,
            'size'   : '15'}
    plt.rc('font',**font)
    ax=plt.subplot(1,1,1)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    om1, om2 = (min(omi_mesh)+omi_shift)/1000.0, (max(omi_mesh)+omi_shift)/1000.0
    los1, los2 = min(eloss_mesh), max(eloss_mesh)
    plt.ylim([om2, om1])
    plt.xlim([los1, los2])
    cax=plt.imshow(np.transpose(rixs[0]+rixs[1]), extent=[los1, los2, om2, om1], origin='upper', aspect='auto',cmap='jet', interpolation='bicubic')
    plt.plot(eloss_mesh, [11.215]*len(eloss_mesh), '--', color='white')
    plt.xlabel(r"Energy loss (eV)")
    plt.ylabel(r"Incident energy (keV)")
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.12, top=0.95,  wspace=0.05, hspace=0.02)
    plt.savefig("rixs_map.pdf")
