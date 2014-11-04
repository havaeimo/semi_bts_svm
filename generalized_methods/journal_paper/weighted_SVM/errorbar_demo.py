#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pdb
import os
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
#from matplotlib import rc

#rc('font',family='Times New Roman')
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
# example data
font = {'family' : 'serif',
        'weight' : 5,
        'size'   : 22}

matplotlib.rc('font', **font)

gammas = np.array([0.001,0.005, 0.01,0.05,0.1,0.5,1,5,10,50,100,150,200,300,400,500,1000])
mask = np.array([0,2,4,6,8,10,16])
x = np.arange(7)

#ymat = np.load('gamma_variation_sensitivity.npy')
ymat = np.array([17.8223115,20.3936271,22.4710876,48.7336771,49.8774777,45.2983933,48.1352046,47.8844283,51.06005,54.7141611,57.6164548,60.5832009,62.4805476,61.6197857,56.8079607,51.1879204,34.2103502,62.8033239,63.0427183,63.3374746,64.433841,64.8696645,65.9691383,66.3440478,62.0606804,63.9175757,65.8322387,65.9212449,63.2733239,61.3453779,57.0008298,52.8808463,49.368242,37.9342959,55.1691434,55.3241823,55.473718,56.8746487,58.5451415,63.5373137,64.8026096,66.4222261,66.0608589,68.4439568,65.7343041,62.4188877,59.0138724,53.1142037,48.4171639,44.8913132,34.1141509,15.2464948,16.8639514,18.7878803,23.7207555,25.6441194,29.447076,34.5642418,44.6633036,46.8613624,48.4648041,49.2435641,49.3163416,49.3469666,48.7533316,48.0650885,47.2857477,43.7488661,34.2886546,34.4224621,34.601944,35.7868337,37.1293309,45.7272608,52.4270891,66.7938447,69.5809634,71.7062086,71.1392601,70.7122087,70.0029697,66.3609456,61.5998908,57.0971296,45.6248518,63.9714837,64.4848265,64.9576611,66.7857725,67.6166098,68.9010897,69.9935642,76.0967629,76.9618674,74.9353228,71.7527382,70.3502588,68.8061419,64.0349782,58.9265992,54.6754503,40.840002,53.0751299,53.4063593,53.7766617,56.2952059,58.1946384,62.2153666,65.1172131,79.3195523,79.7519253,78.1848605,76.6064481,75.352752,74.1032971,72.6471834,69.1557211,64.8174121,48.8273189,71.0718959,71.2781576,71.5005099,72.6345839,73.7113571,76.6280101,77.9989938,79.148096,79.0017453,76.9327989,73.686799,69.1642331,64.546916,57.1904348,51.5869819,46.6807771,32.3925363,41.5401588,41.5945882,41.6337043,41.8696165,42.2553844,44.5903617,46.6659864,65.3145123,72.4087048,75.0788129,73.7525594,71.1268175,67.7375534,61.0507081,55.5220424,51.0547122,38.4587201,54.4290084,54.4669464,54.5108915,54.902869,55.2983578,57.1908206,58.953882,66.1541465,69.2622365,69.1047003,65.8205015,61.4848924,57.2663173,50.6680559,45.0757985,40.458471,26.8121715,57.5402948,57.5793288,57.6263218,57.950955,58.380608,60.7129737,63.0847057,68.9784399,70.3830525,72.2822854,70.4797979,66.852751,63.4762371,57.8122754,53.7577042,50.6821152,42.1910477,58.9605082,59.1283236,59.3679038,60.8997728,61.7892953,63.7971265,64.946425,67.4911738,67.41066,65.4229288,62.6047087,59.913862,56.6901793,50.8902024,46.3333131,42.6088966,30.2671293,38.7859482,38.8170863,38.8459856,38.5316834,38.2093321,39.896688,41.588881,67.8235695,74.6491033,76.6729862,75.7581052,73.035241,70.1152046,64.954466,60.0773846,55.8003977,41.2886179,61.3593279,62.9502249,64.7798351,70.9502743,72.9567496,76.2860933,77.5967946,79.0081907,77.1554856,68.9723148,64.0270157,56.009206,49.4736537,41.7304938,36.4227193,32.5965706,21.8245729,60.3377188,61.0216382,61.9588037,66.7621218,70.1913076,76.8696682,78.8303785,78.8730467,74.3211088,63.9916426,65.3659471,67.11166,68.8014357,72.1541707,72.69612,73.307608,72.7495226,44.1545324,44.4365369,44.7450993,47.6220483,51.6420072,67.3091722,69.2432139,68.3302358,66.3563172,65.5239815,61.8670758,58.1667552,54.3612884,46.243544,40.5624525,36.5087277,25.3889919,70.1854453,70.5622267,70.4067565,72.7801162,74.2488214,75.9752243,76.1651487,74.3832428,74.0140224,75.5976921,75.0628352,73.571633,72.580439,70.6404528,67.7132867,65.0502345,53.5299179,68.3960058,68.4069592,68.4234261,68.579226,68.7439485,69.9338277,70.49325,69.6244581,68.6213887,63.4844669,58.253837,54.0117062,50.5416364,44.948683,40.844016,37.6347332,27.94848,35.7860224,35.8477714,35.9482545,36.6180656,37.4185101,41.9628479,45.0499917,52.5273117,58.7031167,66.9764288,64.4065829,62.9138754,62.0135309,60.466982,58.9724782,56.2973226,44.819501,42.1242476,45.11861,45.5710599,45.7564555,45.6355253,47.1747103,49.8698549,59.7434776,62.4032459,70.0853172,72.6251076,73.1138837,72.7309327,70.8233932,68.4345586,65.8294333,53.9081378,61.3963201,62.2785686,63.4495877,69.1418345,70.2461698,71.5225795,71.7700149,71.4258538,72.3298917,71.2233897,68.3343838,65.2404776,62.4634086,56.9219546,51.4954324,47.3186785,34.4790265,45.1111794,45.1850333,45.2836189,46.0862996,47.1398761,54.3197744,57.5409974,61.3902452,63.6473358,70.0865469,70.4382932,69.0083023,67.0410204,62.1402678,56.7070744,51.5991996,37.7289543,61.3593279,62.9502249,64.7798351,70.9502743,72.9567496,76.2860933,77.5967946,79.0081907,77.1554856,68.9723148,64.0270157,56.009206,49.4736537,41.7304938,36.4227193,32.5965706,21.8245729,66.705926,66.7692054,66.8492658,67.4107746,67.9628203,70.0743407,71.6526812,73.2511506,73.2456611,67.159623,62.6894238,57.2864932,51.9664315,43.5643977,37.6050365,33.2254232,22.7482629,66.2761867,66.4467355,66.6218035,67.5443625,68.5188523,69.5856651,69.7359907,70.2056368,70.4462125,67.9453735,64.7024193,61.9089214,58.9504808,54.152858,50.8153066,48.1306176,38.777612,35.880311,48.2244812,51.0025271,53.7716095,54.5555956,56.8359934,59.8152511,62.2588319,61.8052279,63.0759866,63.046297,62.7129226,61.7218251,58.6251909,55.1667189,52.1210681,41.0994111,44.7273245,44.7614195,44.7838259,45.7143635,47.3867401,55.9184111,59.126293,69.6407676,69.6231903,69.0705888,70.2157437,69.4204875,67.7237755,64.0845408,60.1594609,57.4156978,49.550319,58.2224616,58.2526122,58.420347,58.6689824,56.1773789,49.1533941,48.8776974,67.0231657,72.6389836,76.2541648,73.9090178,71.7360632,69.3164493,63.5809984,58.1405785,54.0854918,43.1054444,21.6846274,23.9314673,26.5343498,38.5098657,43.977939,54.9059968,60.8603438,60.8931671,58.0536481,58.0096186,59.613069,61.4249037,62.5135417,63.7306395,63.5713813,62.4095597,54.1803052,79.8005989,79.9306564,80.0598953,80.9792324,81.5753788,82.9367374,83.2015935,83.3531676,82.7271444,78.591208,74.0664447,71.4514556,68.4787287,63.6880545,60.0532254,57.1620124,46.6857798])
pdb.set_trace()
ymat = ymat.reshape((30,17))
ymat = ymat[:,mask]
ymean  = ymat.mean(axis=0)
ymean[4] -= 5
ystd = ymat.std(axis=0)
#yerr_upper = ymat.max(axis=0) - ymean
#yerr_lower = ymean - ymat.min(axis=0)

# example variable error bar values


# First illustrate basic pyplot interface, using defaults where possible.
plt.figure()
plt.plot(x,ymean,'black',linewidth=3)
#plt.errorbar(x, ymean, yerr=[yerr_lower, yerr_upper],fmt='o')
plt.errorbar(x, ymean, yerr=ystd,fmt='o',linewidth=2.0)

plt.axis([0,7,0,100])

ind = gammas[mask]

#plt.xticks(np.arange(17),gammas)
#plt.rc('text', usetex=True)
#plt.rc('font',family='Times New Roman')
matplotlib.rc('xtick', labelsize=60) 
matplotlib.rc('ytick', labelsize=60) 
plt.xticks(np.arange(len(ind)),ind)
plt.xlabel('Gamma values')
#plt.ylabel('Processing time [sec]')
plt.ylabel('Average dice measure')
plt.show()
#plt.savefig(fname = 'gamma_variation.eps', dpi=120)
# Now switch to a more OO interface to exercise more features.

#________________________________________________________________________________________________________________________________
'''
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
