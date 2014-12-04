import numpy as np
import os
import sys
import fcntl
import copy
import pdb
sys.path.append('/home/local/USHERBROOKE/havm2701/git.repos/semi_bts_svm/semi_bts_svm/generalized_methods/')
sys.path.append('/home/local/USHERBROOKE/havm2701/git.repos/semi_bts_svm/semi_bts_svm/generalized_methods/journal_paper/')
from string import Template
import mlpython.datasets.store as dataset_store
from mlpython.learners.third_party.libsvm.classification import SVMClassifier
import compute_statistics
import time
import data_utils
from model_svm_6d_hyperparameter_Sensitivity import svm_model as svm_model
from model_svm_6d_hyperparameter_Sensitivity import load_data as load_data
#import ipdb

# make a dictionary of the selected brains with their best c and gamma parameters. 
brain_list = { 'HG_0022': [1,5], 'HG_0025': [1,50]}
sys.argv.pop(0); # Remove first argument

# Get arguments
dataset_directory = sys.argv[0]
#dataset_name = sys.argv[1]
output_folder = sys.argv[1]

results_path = output_folder + '/libsvm_results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

#gammas = [0.001,0.005, 0.01,0.05,0.1,0.5,1,5,10,50,100,150,200,300,400,500,1000]
#Cs = [1,5,10,25,50,75,100,150,200,250,300,400,500,750,1000,1250,1500]
noise_factor = [ 5,10,15,20,25,30,35,40,45,50,55,60,65,70]
noise_factor = np.array(noise_factor)
noise_factor = (noise_factor*1.0) / 100
# measure the sensitivity of gamma for the selected brains and save the text file

brain_names = brain_list.keys()
results_file_c = 'libsvm_measures_C.txt'
results_file_g = 'libsvm_noiseres_noise_factort'
for brain in brain_names:
    datasets = load_data(dataset_directory , brain)
    resultg1, resultg2 = '' ,''
    C = brain_list[brain][0]
    brain_str = brain + '\t' + 'C=' + str(C) + ' \n'
    gamma = brain_list[brain][1]
    params = ['rbf', 3, gamma, 0, C]
    dice , processed_time = svm_model(dataset_directory, brain, params, datasets)
    for noise in noise_factor:
        gamma_noisy = np.abs(gamma + (gamma *(noise * np.random.randn())))
        params_noisy = ['rbf', 3, gamma_noisy, 0, C]
        dice_noisy , processed_timeg = svm_model(dataset_directory, brain, params_noisy, datasets)
        dice_g = np.abs(dice - dice_noisy)
        resultg1 += "%.4f" % dice_g + '\t'
        resultg2 += "%.4f" % processed_timeg + '\t'
    
    resultg1 += '\n'
    resultg2 += '\n'
     
    if not os.path.exists(results_path + results_file_g):
          with open(results_path + results_file_g,'w') as g:
            g.write(brain_str)
            g.write(resultg1)
            g.write(resultg2)
    else:
          with open(results_path + results_file_g,'a') as g:
              g.write(brain_str)
              g.write(resultg1)
              g.write(resultg2) 
'''
for brain in brain_names:
    datasets = load_data(dataset_directory , brain)
    resultc1, resultc2 = '' ,''           
    gamma = brain_list[brain][1]
    brain_str = brain + '\t' + 'gamma=' + str(gamma) + ' \n'
    
    for C in Cs:
        paramsc = ['rbf', 3, gamma, 0, C]
        dice_c , processed_timec = svm_model(dataset_directory, brain, paramsc, datasets)
        pdb.set_trace()
        #if brain == 'LG_0008':
        #	pdb.set_trace()

        resultc1 += "%.7f" % dice_c + '\t'
        resultc2 += "%.4f" % processed_timec + '\t'
    resultc1 += '\n'
    resultc2 += '\n'
     
    if not os.path.exists(results_path + results_file_c):
          with open(results_path + results_file_c,'w') as c:
            c.write(brain_str)
            c.write(resultc1)
            c.write(resultc2)
    else:
          with open(results_path + results_file_c,'a') as c:
              c.write(brain_str)
              c.write(resultc1)
              c.write(resultc2)
'''
