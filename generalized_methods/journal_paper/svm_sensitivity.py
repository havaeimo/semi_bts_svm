import numpy as np
import os
import sys
import fcntl
import copy
import pdb
sys.path.append('/home/local/USHERBROOKE/havm2701/git.repos/semi_bts_svm/semi_bts_svm/generalized_methods/')
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
brain_list = {'HG_0002': [1,50], 'HG_0001': [1,0.01], 'HG_0003': [1,1], 'HG_0010': [1,50], 'HG_0008': [1,5], 'HG_0012': [50,50], 'HG_0011': [1,5], 'HG_0022': [1,5], 'HG_0025': [1,50], 'HG_0027': [1,10], 'LG_0008': [1,200], 'LG_0001': [1,50], 'LG_0006': [1,1], 'LG_0015': [100,200], 'HG_0014': [1,5]}
sys.argv.pop(0); # Remove first argument

# Get arguments
dataset_directory = sys.argv[0]
#dataset_name = sys.argv[1]
output_folder = sys.argv[1]

results_path = output_folder + '/libsvm_results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

gammas = [0.001,0.005, 0.01,0.05,0.1,0.5,1,5,10,50,100,150,200,300,400,500,1000]
Cs = [1,5,10,25,50,75,100,150,200,250,300,400,500,750,1000,1250,1500]

# measure the sensitivity of gamma for the selected brains and save the text file

brain_names = brain_list.keys()
results_file_c = 'libsvm_measures_C.txt'
results_file_g = 'libsvm_measures_gamma.txt'

for brain in brain_names:
    datasets = load_data(dataset_directory , brain)
    resultg1, resultg2 = '' ,''
    C = brain_list[brain][0]
    brain_str = brain + '\t' + 'C=' + str(C) + ' \n'
    
    for gamma in gammas:
        paramsg = ['rbf', 3, gamma, 0, C]
        dice_g , processed_timeg = svm_model(dataset_directory, brain, paramsg, datasets)
        
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

for brain in brain_names:
    datasets = load_data(dataset_directory , brain)
    resultc1, resultc2 = '' ,''           
    gamma = brain_list[brain][1]
    brain_str = brain + '\t' + 'gamma=' + str(gamma) + ' \n'
    
    for C in Cs:
        paramsc = ['rbf', 3, gamma, 0, C]
        dice_C , processed_timec = svm_model(dataset_directory, brain, paramsc, datasets)
        #if brain == 'LG_0008':
        #	pdb.set_trace()
        print dice_c
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