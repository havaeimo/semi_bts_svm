#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pdb
# example data
x = [0.001,0.005, 0.01,0.05,0.1,0.5,1,5,10,50,100,150,200,300,400,500,1000]

x = np.arange(17)
#ymat = np.array([0.0441,0.0403,0.0422,0.0326,0.0288,0.0173,0.0115,0.0096,0.0019,0.0000,0.0019,0.0019,0.0019,0.0019,0.0019,0.0019,0.0038,0.0246,0.0246,0.0246,0.0157,0.0134,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0022,0.0470,0.0470,0.0470,0.0470,0.0470,0.0396,0.0371,0.0198,0.0124,0.0074,0.0074,0.0050,0.0025,0.0050,0.0050,0.0050,0.0050,0.0681,0.0545,0.0511,0.0358,0.0341,0.0273,0.0256,0.0034,0.0034,0.0051,0.0034,0.0034,0.0017,0.0034,0.0034,0.0034,0.0051,0.0544,0.0444,0.0464,0.0423,0.0242,0.0101,0.0040,0.0020,0.0020,0.0000,0.0020,0.0020,0.0020,0.0020,0.0020,0.0020,0.0040,0.0274,0.0274,0.0299,0.0274,0.0274,0.0200,0.0175,0.0125,0.0075,0.0000,0.0000,0.0000,0.0000,0.0000,0.0025,0.0025,0.0025,0.0139,0.0139,0.0124,0.0124,0.0108,0.0062,0.0062,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0015,0.0015,0.0031,0.0077, 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0016,0.0016,0.0033,0.0033,0.0049,0.0082,0.1478,0.0447,0.0464,0.0292,0.0120,0.0086,0.0086,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0017,0.0120,0.0262,0.0293,0.0154,0.0123,0.0077,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0343,0.0360,0.0343,0.0206,0.0154,0.0120,0.0103,0.0069,0.0069,0.0051,0.0034,0.0051,0.0051,0.0051,0.0051,0.0069,0.0086,0.0632,0.0620,0.0632,0.0515,0.0456,0.0316,0.0234,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0012,0.0012,0.0047,0.0264,0.0231,0.0214,0.0115,0.0016,0.0016,0.0016,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0556,0.0556,0.0556,0.0225,0.0185,0.0106,0.0040,0.0013,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0013,0.0649,0.0649,0.0649,0.0582,0.0599,0.0483,0.0349,0.0116,0.0033,0.0000,0.0000,0.0000,0.0000,0.0017,0.0033,0.0017,0.0083])
ymat = np.array([37.2976,48.7688,43.6994,56.1524,57.9139,62.0332,64.2871,68.4818,70.6861,65.7860,56.5624,48.2289,42.2894,34.1341,29.3713,26.3973,18.5837,65.0763,64.7310,65.1039,69.2028,69.5647,77.2437,78.0522,79.8578,80.0529,82.1094,79.4803,73.7557,68.7860,62.6456,59.0253,56.7507,45.9253,63.7514,63.7059,63.5656,64.2387,66.1955,67.2723,67.9589,72.3433,73.4106,75.7472,71.0008,65.2291,59.1793,52.2057,50.1191,44.8411,37.7157,49.6032,51.7518,53.3178,48.7453,48.3010,61.9223,72.0553,75.1065,74.5371,75.0967,75.7461,72.8720,68.1121,59.2096,52.3589,48.2909,37.2881,19.6905,29.3190,32.4569,35.5461,37.1889,41.8167,44.0617,48.9054,48.1693,48.4301,49.7819,48.4296,47.4296,45.4472,44.4433,41.7749,38.1690,34.9891,35.5290,36.0195,35.7168,36.3928,42.8944,51.7212,68.8141,72.7905,69.3965,64.8974,59.7561,52.8483,46.7027,42.9364,39.3246,30.4768,72.3762,73.7051,73.7271,74.5835,76.6090,76.9849,77.6326,80.1496,78.8636,73.7459,61.3678,53.1436,47.6924,39.4277,35.4597,31.5340,23.1213,82.5475,82.6898,82.5816,83.0026,82.9975,82.4862,82.7264,84.6295,85.4879,67.4758,48.9670,41.9613,34.7669,28.5106,25.4815,23.7766,17.5100,68.9880,45.8280,57.7505,71.0651,74.3476,73.2935,74.9191,79.0244,79.4854,70.9070,50.5544,36.6726,31.6531,25.6418,22.8905,20.1889,14.3475,56.8820,57.4721,57.9696,61.8074,62.6168,65.4898,66.9990,68.2079,69.6689,70.1855,60.6396,53.8341,48.5618,42.4973,38.9541,36.1401,27.7583,58.7649,59.3962,59.2233,61.9646,63.2334,65.2814,66.4054,68.7993,69.5809,67.7231,61.5489,53.6921,49.1941,42.6910,38.2777,36.3048,28.6245,61.6031,61.5002,61.7983,64.6623,65.6451,67.0970,68.0619,76.5894,76.7518,77.8469,73.3837,66.4788,59.3723,48.5912,43.3243,39.1426,30.8886,47.7517,47.6582,49.1099,54.5919,56.9654,64.7015,69.3295,79.5861,81.0293,80.0948,70.4738,60.7355,55.9251,45.6166,38.7673,35.1082,26.9038,53.7049,53.7184,53.8028,57.3445,58.2642,61.1816,64.4576,67.3271,67.2660,62.9434,53.7504,46.1622,42.6775,36.1981,31.9083,30.0451,24.1767,37.6797,37.0681,36.9393,36.2922,37.0368,48.2526,55.1251,72.3533,71.7998,75.3379,71.1726,65.5864,62.4081,56.5632,51.4036,47.3266,35.5905])

ymat = ymat.reshape((15,17))
ymean  = ymat.mean(axis=0)

yerr_upper = ymat.max(axis=0)
yerr_lower = ymat.min(axis=0)

# example variable error bar values


# First illustrate basic pyplot interface, using defaults where possible.
plt.figure()
plt.errorbar(x, ymean, yerr=[yerr_lower, yerr_upper],fmt='o')
#plt.axis([min(x),max(x),0,100])
plt.axis('normal')
plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")
plt.show()
# Now switch to a more OO interface to exercise more features.