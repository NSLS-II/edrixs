#!/usr/bin/env python

import numpy as np

data=np.loadtxt('rixs.dat')

omega=data[0:100,1]
eloss=data[0::100,0]

rixs = np.zeros((4, 1000,100), dtype=np.float64)
rixs[0] = data[:,2].reshape((1000,100))
rixs[1] = data[:,3].reshape((1000,100))
rixs[2] = data[:,4].reshape((1000,100))
rixs[3] = data[:,5].reshape((1000,100))

f=open('rixs_1.dat', 'w')
f.write("# 852.91414141   857.4")
for i in range(1000):
    f.write("{:20.10f}{:20.10f}{:20.10f}{:20.10f}{:20.10f}\n".format(eloss[i], rixs[0,i,28], rixs[1,i,28], rixs[2,i,28], rixs[3,i,28]))
f.close()

f=open('rixs_2.dat', 'w')
f.write("# 854.78282828   857.4")
for i in range(1000):
    f.write("{:20.10f}{:20.10f}{:20.10f}{:20.10f}{:20.10f}\n".format(eloss[i], rixs[0,i,65], rixs[1,i,65], rixs[2,i,65], rixs[3,i,65]))
f.close()

