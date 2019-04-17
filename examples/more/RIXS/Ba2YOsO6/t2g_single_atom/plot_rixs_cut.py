#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': '25'}

plt.rc('font', **font)

plt.figure(figsize=(16, 8))

data = np.loadtxt('rixs.dat')

nomega, nloss = 100, 1000

eloss = data[0::nomega, 0]
omega = data[0:nomega, 1]

spectra_pi_pol = data[:, 2].reshape((nloss, nomega)) + data[:, 3].reshape((nloss, nomega))
spectra_sigma_pol = data[:, 4].reshape((nloss, nomega)) + data[:, 5].reshape((nloss, nomega))

ax = plt.subplot(1, 2, 1)
plt.xlim((-0.5, 2.0))
plt.ylim((0, 1.5))
plt.plot(eloss, spectra_pi_pol[:, 25] * 50, linewidth=2, label="10.874 keV")

plt.xlabel(r'Energy loss (eV)')
plt.ylabel(r'RIXS Intensity (a.u.)')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.tick_params(axis='x', which='major', length=10, width=3)
ax.tick_params(axis='x', which='minor', length=5, width=3)
ax.tick_params(axis='y', which='major', length=10, width=3)
ax.tick_params(axis='y', which='minor', length=5, width=3)
plt.legend(
    loc=1,
    ncol=1,
    frameon=False,
    borderpad=0.2,
    columnspacing=0.5,
    handlelength=1.8,
    handletextpad=0.4,
    fontsize=25)
plt.text(1, 1.3, r"$\pi$-pol", fontsize=30)

ax2 = plt.subplot(1, 2, 2)
plt.xlim((-0.5, 2))
plt.ylim((0, 1.5))
plt.plot(eloss, spectra_sigma_pol[:, 25] * 50, linewidth=2, label="10.874 keV")
plt.xlabel(r'Energy loss (eV)')
plt.ylabel(r'RIXS Intensity (a.u.)')
ax2.xaxis.set_major_locator(MultipleLocator(1))
ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
ax2.tick_params(axis='x', which='major', length=10, width=3)
ax2.tick_params(axis='x', which='minor', length=5, width=3)
ax2.tick_params(axis='y', which='major', length=10, width=3)
ax2.tick_params(axis='y', which='minor', length=5, width=3)
plt.legend(
    loc=1,
    ncol=1,
    frameon=False,
    borderpad=0.2,
    columnspacing=0.5,
    handlelength=1.8,
    handletextpad=0.4,
    fontsize=25)
plt.text(1, 1.3, r"$\sigma$-pol", fontsize=30)

plt.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.95, wspace=0.2, hspace=0.25)

plt.savefig("rixs_cut.pdf")

print("Job Done !")
