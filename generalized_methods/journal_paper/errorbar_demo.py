#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pdb
# example data
'''
gammas = [0.001,0.005, 0.01,0.05,0.1,0.5,1,5,10,50,100,150,200,300,400,500,1000]

x = np.arange(17)

ymat = np.load('gamma_variation_sensitivity.npy')
ymat = ymat.reshape((15,17))
ymean  = ymat.mean(axis=0)

yerr_upper = ymat.max(axis=0) - ymean
yerr_lower = ymean - ymat.min(axis=0)

# example variable error bar values


# First illustrate basic pyplot interface, using defaults where possible.
plt.figure()
plt.errorbar(x, ymean, yerr=[yerr_lower, yerr_upper],fmt='o')

plt.axis([0,17,0,100])
plt.xticks(np.arange(17),gammas)

plt.xlabel('gamma values')
plt.ylabel('Average dice measure')
plt.show()
#plt.savefig(fname = 'gamma_variation.eps', dpi=120)
# Now switch to a more OO interface to exercise more features.
'''
'''
#________________________________________________________________________________________________________________________________

ymat = np.load('c_variation_sensitivity.npy')
Cs = [1,5,10,25,50,75,100,150,200,250,300,400,500,750,1000,1250,1500]
ymat = ymat.reshape((15,17))
ymean  = ymat.mean(axis=0)
x = np.arange(17)
yerr_upper = ymat.max(axis=0) - ymean
yerr_lower = ymean - ymat.min(axis=0)

# example variable error bar values


# First illustrate basic pyplot interface, using defaults where possible.
plt.figure()
plt.errorbar(x, ymean, yerr=[yerr_lower, yerr_upper],fmt='o')

plt.axis([0,17,0,100])
plt.xticks(np.arange(17),Cs)

plt.xlabel('C values')
plt.ylabel('Average dice measure')
plt.show()
#plt.savefig(fname = 'C_variation.eps', dpi=120)
'''
#------------------------------------------------------------------------------------------------------------------------------------
gammas = [0.001,0.005, 0.01,0.05,0.1,0.5,1,5,10,50,100,150,200,300,400,500,1000]

x = np.arange(17)

ymat = np.array([94.58,94.61,94.19,94.21,93.96,82.32,77.17,70.06,67.39,65.57,70.26,74.53,77.46,82.91,87.18,91.14,103.87,113.56,115.21,114.53,100.89,95.76,81.66,76.09,66.8,65.73,67.56,72.96,77.35,80.3,84.43,89.41,92.52,104,119.9,119.94,120.61,120.23,117.48,103.23,96.81,85.96,81.37,79.92,85.1,88.96,92.06,97.23,102.16,105.83,121.39,114.09,99.1,94.86,85.1,82.6,77.23,74.48,70.52,68.1,72.73,80.42,86.93,92.21,97.9,104.1,109.54,129.14,118.26,99.38,95.56,89.31,86.61,81.97,80.86,72.88,72.07,78.67,87.54,93.44,97.14,104.66,110.07,114.7,132.54,98.1,99.63,99.28,98.99,99.8,91.19,87.54,80.14,78.78,83.86,91.51,97.29,101.73,106.74,111.92,115.58,129.63,153.54,151.04,132.2,106.47,100.19,82.17,76.15,68.86,66.7,72.23,77.82,83.09,87.06,95.7,101.3,107.09,125.88,129.46,129.86,129.89,114.65,101.79,79.89,75.91,69.34,69.94,78.39,87.72,94.74,100.94,107.59,113.77,118.21,139.63,139.25,139.88,139.9,137.12,132.16,111.22,100.84,84.75,81.99,90.66,98.94,105.92,111.22,119.53,126.07,131.66,151.62,177.62,178.25,179.94,146.86,133.58,114.81,105.55,90.53,86.63,95.13,103.75,109.32,115.65,124.15,132.72,139.4,168.12,130.32,131.12,130.98,115.48,109.45,94.11,88.53,80,78.58,86.63,98.3,104.56,110.93,119.66,128.25,133.98,155.79,211.52,212,206.26,161.25,148.38,124.52,114.89,91.08,84.64,89.2,99.07,107.28,112.87,123.89,131.61,139.56,172.22,161.13,160.88,150.37,126.61,116.67,93.01,86.21,76.36,74.28,82.14,92.34,98.25,103.65,111.96,119.93,124.72,144.25,189.84,191.15,189.79,156.59,139.23,111.71,101.48,84.52,80.76,84.83,93.54,100.29,107.41,116.78,126.66,136.28,170.38,140.18,140.74,141.22,133.2,128.97,111.94,106.41,91.9,86.37,89.46,97.6,103.92,108.73,116.7,123.83,130.19,149.86])
np.save('gamma_variation_time_sensitivity.npy',ymat)
ymat = ymat.reshape((15,17))
ymean  = ymat.mean(axis=0)

yerr_upper = ymat.max(axis=0) - ymean
yerr_lower = ymean - ymat.min(axis=0)

# example variable error bar values

# First illustrate basic pyplot interface, using defaults where possible.
plt.figure()
plt.errorbar(x, ymean, yerr=[yerr_lower, yerr_upper],fmt='s', mfc='black', ms=5, ecolor='black')
# marker='s'  use this argument to connect the markers to gether while mfc produces scatter plot of errorbars. 
# mfc is used to determince the color of marker
# ecolor is used for the color of the error bars

plt.axis([0,17,50,250])
plt.xticks(np.arange(17),gammas)

plt.xlabel('gamma values')
plt.ylabel('Average time measure in sec')
plt.show()
