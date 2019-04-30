#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': '20'}


plt.rc('font', **font)
# fig=plt.figure(figsize=(20, 15))

data = np.loadtxt('rixs.dat')
nomega, nloss = 21, 1000
eloss = data[0::nomega, 0]
omega = data[0:nomega, 1] / 1000.0
spectrum = np.transpose(data[:, 2].reshape((nloss, nomega)) + data[:, 3].reshape((nloss, nomega)))

ax = plt.subplot(1, 1, 1)
plt.ylim([max(omega), min(omega)])
plt.xlim([min(eloss), max(eloss)])
cax = plt.imshow(
    spectrum,
    extent=[
        min(eloss),
        max(eloss),
        max(omega),
        min(omega)],
    origin='upper',
    aspect='auto',
    cmap='jet',
    interpolation='bicubic')
plt.plot(eloss, [11.215] * len(eloss), '--', color='white')

ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.001))
ax.yaxis.set_minor_locator(MultipleLocator(0.0005))

ax.tick_params(axis='x', which='major', length=6, width=1.5)
ax.tick_params(axis='x', which='minor', length=3, width=1.5)
ax.tick_params(axis='y', which='major', length=6, width=1.5)
ax.tick_params(axis='y', which='minor', length=3, width=1.5)

plt.xlabel(r"Energy loss (eV)")
plt.ylabel(r"Incident energy (keV)")

plt.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.95, wspace=0.05, hspace=0.02)

plt.savefig("rixs_map.pdf")
# plt.show()

print("Job Done !")
