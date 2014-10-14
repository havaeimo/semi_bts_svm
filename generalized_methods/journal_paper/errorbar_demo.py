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
Factors = [1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8]

x = np.arange(len(Factors))

ymat = np.array([66.3082867,67.2302335,69.0286384,67.9647716,67.826511,66.6569785,67.9769242,64.641953,67.115671,71.1176489,63.66821,68.1869923,77.5962828,78.3798159,77.653164,76.8018755,77.6166628,77.7040143,76.8789181,76.9822891,76.281492,78.4217981,78.1233291,77.7036313,70.3082279,66.615634,70.6590427,67.9426893,66.9907355,65.3702092,64.3942987,63.1107797,73.204839,72.6936391,60.2448906,69.9217453,76.2832908,75.7251498,74.753151,54.6667794,76.6837057,74.9660265,68.1140826,75.4714654,72.7047556,76.2490864,33.7983031,77.1977692,49.6127019,44.3718146,61.0275823,52.1120263,51.6305415,47.5180978,51.7384999,37.0527218,54.4869466,28.3593989,37.0649599,42.2482922,62.5308659,62.7339932,65.8783202,59.1214273,60.4734516,59.4228967,54.8129001,63.4894852,49.9211832,61.223264,65.0185133,47.6026752,77.816917,78.4031158,76.7092936,77.5127312,79.3224531,76.1368663,75.9034961,72.122051,77.3663953,73.1202462,74.4397853,73.1928392,80.1884351,80.3687029,79.3153366,80.4281812,80.1709555,80.6149763,79.4149562,79.9458606,79.7543992,79.7189252,78.4510304,78.2384909,78.7850759,78.3285214,77.2280844,77.2146799,70.425883,77.9468235,72.0646232,73.1177085,78.9594316,73.3087341,70.1173185,75.8279428,67.1927649,66.6706187,69.301429,66.2575998,67.1845464,67.5823456,66.383472,65.0844509,65.8095312,61.7976706,67.1798486,66.8001399,63.8577186,65.2369039,65.0079005,64.7164926,63.4282308,63.7248674,60.6226586,63.3241952,60.2179949,58.5549679,64.1343831,61.9702052,74.3367428,75.0373254,75.0591057,75.8015575,74.83692,76.9640368,74.6608012,72.5467787,75.2116991,69.8014193,74.1445943,73.3808986,67.8620721,63.1804205,60.5651934,63.7273403,60.8080466,60.8027918,61.1927626,59.2896006,54.5690125,56.8551822,60.6099602,52.7895263,66.5005599,68.2620293,64.2754856,64.3855235,64.8436301,65.7092219,63.7409836,66.8361522,62.7724064,58.6173797,59.5298234,58.2495432,69.6956654,67.7195614,70.4636127,70.6814256,67.8051306,68.5281757,72.0480422,71.4720669,67.2416215,66.0262525,67.9690222,67.9218263])
np.save('gamma_variation_time_sensitivity.npy',ymat)
#pdb.set_trace()
ymat = ymat.reshape((15,len(Factors)))
ymean  = ymat.mean(axis=0)

yerr_upper = ymat.max(axis=0) - ymean
yerr_lower = ymean - ymat.min(axis=0)

# example variable error bar values

# First illustrate basic pyplot interface, using defaults where possible.
plt.figure()
#plt.errorbar(x, ymean, yerr=[yerr_lower, yerr_upper],fmt='s', mfc='black', ms=5, ecolor='black')
# marker='s'  use this argument to connect the markers to gether while mfc produces scatter plot of errorbars. 
# mfc is used to determince the color of marker
# ecolor is used for the color of the error bars
for i in range(15):
 plt.plot(ymat[i,:])

plt.axis([0,12,0,100])
plt.xticks(np.arange(17),Factors)

plt.xlabel('gamma values')
plt.ylabel('Average time measure in sec')
plt.show()
