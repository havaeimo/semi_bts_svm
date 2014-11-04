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
from model_svm_6d_hyperparameter_Sensitivity import svm_model2 as svm_model2
#import ipdb

# make a dictionary of the selected brains with their best c and gamma parameters. 
#brain_list = {'HG_0002': [1,50], 'HG_0001': [1,0.01], 'HG_0003': [1,1], 'HG_0010': [1,50], 'HG_0008': [1,5], 'HG_0012': [50,50], 'HG_0011': [1,5], 'HG_0022': [1,5], 'HG_0025': [1,50], 'HG_0027': [1,10], 'LG_0008': [1,200], 'LG_0001': [1,50], 'LG_0006': [1,1], 'LG_0015': [100,200], 'HG_0014': [1,5]}
#'HG_0002': [1,50], 'HG_0001': [1,0.01], 'HG_0003': [1,1], 'HG_0004': [1500,10], 'HG_0005': [1,5], 'HG_0006': [1,200], 'HG_0007': [1500,500], 'HG_0009': [1,5], 'HG_0010': [1,50], 'HG_0008': [1,5], 'HG_0012': [50,50], 'HG_0013': [1,5], 'HG_0011': [1,5], 'HG_0014': [1,5], 'HG_0015': [1,5], 'HG_0022': [1,5], 'HG_0024': [1,50], 'HG_0025': [1,50], 'HG_0026': [50,10], 'HG_0027': [1,10], 'LG_0001': [1,50], 'LG_0002': [10,50], 'LG_0004': [1,100], 'LG_0008': [1,200], 'LG_0006': [1,1], 'LG_0011': [1,1], 'LG_0012': [50,100], 'LG_0014': [1,100], 'LG_0013': [1,1], 'LG_0015': [100,200]
brain_list = {'HG_0026': [50,10],  'LG_0002': [10,50], 'LG_0004': [1,100]}
sys.argv.pop(0); # Remove first argument

# Get arguments
dataset_directory = sys.argv[0]
#dataset_name = sys.argv[1]
output_folder = sys.argv[1]

results_path = output_folder + '/libsvm_results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

gammaps = [0.001,0.005, 0.01,0.05,0.1,0.5,1,5,10,50,100,150,200,300,400,500,1000]
#Cs = [1,5,10,25,50,75,100,150,200,250,300,400,500,750,1000,1250,1500]

# measure the sensitivity of gamma for the selected brains and save the text file

brain_names = brain_list.keys()
results_file_c = 'libsvm_measures_C.txt'
results_file_g = 'libsvm_measures_gamma.txt'

for brain in brain_names:
    datasets = load_data(dataset_directory , brain)
    resultc1, resultc2 = '' ,''           
    gamma = brain_list[brain][1]
    C = brain_list[brain][0]
    brain_str = brain + '\t' + 'gamma=' + str(gamma) + ', \t'
    brain_str +=  'C=' + str(C) + ', \n'
    for gammap in gammaps:
        paramsc = [ gamma, gammap,C]
        dice_c , processed_timec = svm_model2( paramsc, datasets)
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
