#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

font = {'family' : 'Times New Roman' ,
	    'weight' : 'normal' ,
		'size'   : '25'}

plt.rc('font',**font)

plt.figure(figsize=(16,8))

data=np.loadtxt('rixs.dat')

nomega, nloss=100,1000

eloss = data[0::nomega,0]
omega = data[0:nomega,1]

spectra_pi_pol = data[:,2].reshape((nloss, nomega)) + data[:,3].reshape((nloss, nomega))
spectra_sigma_pol = data[:,4].reshape((nloss, nomega)) + data[:,5].reshape((nloss, nomega))

plt.ylim([min(omega),max(omega)])
plt.xlim([min(eloss),max(eloss)])

ax=plt.subplot(1,2,1)
plt.imshow(spectra_pi_pol,    extent=[min(omega),max(omega), min(eloss), max(eloss)], origin='lower', aspect='auto',cmap='rainbow', interpolation='gaussian')
plt.ylabel(r'Energy loss (eV)')
plt.xlabel(r'Energy of incident photon (eV)')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.tick_params(axis='x', which='major', length=10, width=3)
ax.tick_params(axis='x', which='minor', length=5,  width=3)
ax.tick_params(axis='y', which='major', length=10, width=3)
ax.tick_params(axis='y', which='minor', length=5,  width=3)

ax2=plt.subplot(1,2,2)
plt.imshow(spectra_sigma_pol, extent=[min(omega),max(omega), min(eloss), max(eloss)], origin='lower', aspect='auto',cmap='rainbow', interpolation='gaussian')
#plt.colorbar()
#plt.ylabel(r'Energy loss (eV)')
plt.xlabel(r'Energy of incident photon (eV)')
ax2.xaxis.set_major_locator(MultipleLocator(1))
ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
ax2.yaxis.set_major_locator(MultipleLocator(1))
ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
ax2.tick_params(axis='x', which='major', length=10, width=3)
ax2.tick_params(axis='x', which='minor', length=5 , width=3)
ax2.tick_params(axis='y', which='major', length=10, width=3)
ax2.tick_params(axis='y', which='minor', length=5 , width=3)


plt.tight_layout(pad=0.8)

plt.savefig("rixs.pdf")
#plt.show()

print("Job Done !")
