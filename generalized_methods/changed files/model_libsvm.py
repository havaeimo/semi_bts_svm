import numpy as np
import os
import sys
import fcntl
import copy
from string import Template
import mlpython.datasets.store as dataset_store
from mlpython.learners.third_party.libsvm.classification import SVMClassifier
import compute_statistics
import time
import data_utils
import ipdb

sys.argv.pop(0);	# Remove first argument

# Check if every option(s) from parent's script are here.

arg_length = len(sys.argv)
if arg_length not in [3,5,6]:
    print len(sys.argv)
    print "Usage: python model_libsvm.py mlpython_dataset_directory dataset_name output_folder [train_filename test_filename [background_filename]]"
    print "Filenames for the train/test/background files are necessary on the first use, they are not needed later on, as trainset/validset/testset files are created"
    print "Ex.: python model_libsvm.py brains_data/Brats-2_training/ HG_0001 . interaction.txt allpoints.txt background.txt"
    sys.exit()


# Get arguments
dataset_directory = sys.argv[0]
dataset_name = sys.argv[1]
output_folder = sys.argv[2]
train_filename = None
test_filename = None
background_filename = None
use_spatial_dim = 1;

if arg_length >= 4:
    train_filename = sys.argv[3]
    test_filename = sys.argv[4]
    
    if arg_length == 6:
        background_filename = sys.argv[5]
    

input_size = 6 

    
#use_weights = True if sys.argv[1] == 'True' else False
use_weights = False
# Do we want this as command-line parameter?

# Set the constructor
str_ParamOption = ""
str_ParamOptionValue = ""

dataset_dir = None
if dataset_dir is None:
    # Try to find dataset in MLPYTHON_DATASET_REPO
    import os
    repo = os.environ.get('MLPYTHON_DATASET_REPO')
    if repo is None:
        raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
    dataset_dir = os.path.join(os.environ.get('MLPYTHON_DATASET_REPO') + '/' + dataset_directory, dataset_name)

#pdb.set_trace()
# Load data
start_time = time.clock()

all_data = data_utils.load_data(dir_path=dataset_dir, input_size=input_size, train_filename=train_filename, test_filename=test_filename, background_filename=background_filename,load_to_memory=False, use_spatial_dim = 0)


train_data, train_metadata = all_data['train']
valid_data, valid_metadata = all_data['valid']
finaltrain_data, finaltrain_metadata = all_data['finaltrain']
test_data, test_metadata = all_data['test']


import mlpython.mlproblems.classification as mlpb
trainset1 = mlpb.ClassificationProblem(train_data, train_metadata)
validset1 = trainset1.apply_on(valid_data,valid_metadata)
finaltrainset1 = trainset1.apply_on(finaltrain_data,finaltrain_metadata)
testset1 = trainset1.apply_on(test_data,test_metadata)
#

trainset = mlpb.ClassSubsetProblem(trainset1,train_metadata,subset = train_metadata['subset'])
validset = mlpb.ClassSubsetProblem(validset1,valid_metadata,subset = train_metadata['subset'])
testset = mlpb.ClassSubsetProblem(testset1,test_metadata,subset = train_metadata['subset'])
finaltrainset = mlpb.ClassSubsetProblem(finaltrainset1,finaltrain_metadata,subset = train_metadata['subset'])

testset.metadata['input_size'] = 3
trainset.metadata['input_size'] = 3
validset.metadata['input_size'] = 3
finaltrainset.metadata['input_size'] = 3

def compute_error_mean_and_sterror(costs):
    classif_errors = np.hstack(costs)
    classif_mean = classif_errors.mean()
    classif_sterror = classif_errors.std(ddof=1)/np.sqrt(classif_errors.shape[0])

    return classif_mean, classif_sterror


print "Setting hyperparameters gridsearch..."
best_hyperparams = None
best_val_error = np.inf

# SVM model documentation
"""
    -t kernel_type : set type of kernel function (default 2)
        0 -- linear: u'*v
        1 -- polynomial: (gamma*u'*v + coef0)^degree
        2 -- radial basis function: exp(-gamma*|u-v|^2)
        3 -- sigmoid: tanh(gamma*u'*v + coef0)
    -d degree : set degree in kernel function (default 3)
    -g gamma : set gamma in kernel function (default 1/num_features)
    -r coef0 : set coef0 in kernel function (default 0)
    -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
    -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
"""

kernels = ['rbf','sigmoid']
degrees = [1,2,3,4,5,7,10,15]
gammas = [0.01,0.1,1,5,10,50,100,200,500,1000]
coef0s = [-10,-1,-0.1,-0.01,0,0.001,0.01,0.1,1,2,5,10,20]
Cs = [1,10,50,100,500,1000,1500,2000,2500,3000,5000]
'''
kernels = ['rbf']
degrees = [1]
gammas = [0.01]
coef0s = [-10]
Cs = [1]
'''


hyperparams_grid = []
for C in Cs:
    # Linear kernel parameters
    #hyperparams_grid.append(['linear', 3, 1, 0, C])

    # Polynomial kernel parameters
    """
    for gamma in gammas:
        for coef0 in coef0s:
            for degree in degrees:
                hyperparams_grid.append(['polynomial', degree, gamma, coef0, C])
    """
    # Rbf kernel parameters
    for gamma in gammas:
        hyperparams_grid.append(['rbf', 3, gamma, 0, C])

    # Sigmoid kernel parameters
    for gamma in gammas:
        for coef0 in coef0s:
            hyperparams_grid.append(['sigmoid', 3, gamma, coef0, C])

if use_weights:
    label_weights = finaltrainset.metadata['label_weights']
else:
    label_weights = None
    
output_probabilities = True # Or False!

print "Pretraining..."
for params in hyperparams_grid:
    try:
        # Create SVMClassifier with hyper-parameters
        svm = SVMClassifier(shrinking=True, kernel=params[0],degree=params[1],gamma=params[2],coef0=params[3],C=params[4],label_weights=label_weights, output_probabilities=output_probabilities)
    except Exception as inst:
        print "Error while instantiating SVMClassifier (required hyper-parameters are probably missing)"
        print inst
        sys.exit()

    svm.train(trainset)

    outputs, costs = svm.test(validset)
    
    
    errors = compute_error_mean_and_sterror(costs)
    error = errors[0]
    
    if error < best_val_error:
        best_val_error = error
        best_hyperparams = params

print
print 'Classification error on valid set : ' + str(best_val_error)
print
print "Training..."
# Train SVM with best hyperparams on train + validset

best_svm = SVMClassifier(shrinking=True, kernel=best_hyperparams[0],degree=best_hyperparams[1],gamma=best_hyperparams[2],coef0=best_hyperparams[3],C=best_hyperparams[4],label_weights=label_weights, output_probabilities=output_probabilities)

best_svm.train(finaltrainset)

print 'Testing...'

outputs, costs = best_svm.test(testset)



errors = compute_error_mean_and_sterror(costs)
error = errors[0]

print
print 'Classification error on test set : ' + str(error)
print "****************************************"

#ipdb.set_trace()
# Evaluation (compute_statistics.py)
id_to_class = {}
for label, id in testset.class_to_id.iteritems():
    id_to_class[id] = label
 
  
#lbl = np.array([int(data[1]) for data in test_data]) # Ground truth
auto_lbl = np.array([int(id_to_class[output[0]]) for output in outputs]) # Predicted labels

"""
WARNING!
If using probability distributions, we need to remap the ids to the right class (for all outputs)
"""
nb_classes = len(testset.class_to_id)
probabilities = np.zeros((len(testset),5))
for (idx , output) in enumerate(outputs):
    for i in range(nb_classes):
        j = id_to_class[i]
        probabilities[idx,j] = output[i+1]
#ipdb.set_trace()
# Write outputs to disk if result file does not already exist
spatial_file = os.path.join(dataset_dir,'spatialinfo' + '.txt')

spf = open(spatial_file,'r')   



results_path = output_folder + 'libsvm_results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)
if use_weights:
    output_file = results_path + dataset_name + '_weightedlabels_output_txt'
else:
    output_file = results_path + dataset_name + '_libsvm_output.txt'
if not os.path.exists(output_file):
    with open(output_file,'w') as f:
        for label, prob in zip(auto_lbl, probabilities):          
            line = str(label) + ' '
            for i in range(len(prob)): 
                line += str(prob[i]) + ' '
            data = spf.readline()
            data =  data.split()
            line +=  data[0] + ' '
            line +=  data[1] + ' '
            line +=  data[2] + '\n'
            f.write(line)

# Compute processing time
end_time = time.clock()
print "Processing time : " + str(float(end_time - start_time) / 60) + ' minutes'
print
"""
def string_debug(labels):
    result = ''
    for i in range(5):
        result += "# data with label " + str(i) + " : " + str( (labels == i).sum() ).rjust(8) + '\n'
    return result
        
def hyper_to_string(hyperparams):
    result = ''
    result += 'Kernel = ' + str(hyperparams[0]) + ', '
    result += 'C = ' + str(hyperparams[4]) + ', '
    result += 'gamma = ' + str(hyperparams[2]) + ','
    result += 'coef0 = ' + str(hyperparams[3])
    return result
        
def measure_to_string(measure):
    result = ''
    for value in measure:
        result += str(value)[:5].rjust(6)
    return result



result = ''

# Fill prediction / ground truth with zeros for all background points
len_bg = testset.metadata['len_bg']
lbl = np.append(lbl, [0]*len_bg)
auto_lbl = np.append(auto_lbl, [0]*len_bg)

(dice, jaccard, precision, recall) = compute_statistics.compute_eval_multilabel_metrics(auto_lbl, lbl)

lbl_result = string_debug(lbl)
auto_lbl_result = string_debug(auto_lbl)

results_file = 'libsvm_measures.txt'
if not os.path.exists(results_path + results_file):
    result += 'Results = [Edema, non-enhanced tumor, enhanced tumor, complete (abnormality vs healthy)]\n'
    
result += 'Dataset : ' + dataset_name + '\n'
result += 'Model : SVM\n'

result += 'Best hyperparameters : ' + hyper_to_string(best_hyperparams) + '\n'

result += 'Ground truth : \n' + lbl_result
result += 'Prediction : \n' + auto_lbl_result

result += 'Precision = '.ljust(15) + ' '
result += measure_to_string(precision) + '\n'
result += 'Recall = '.ljust(15) + ' '
result += measure_to_string(recall) + '\n'
result += 'Dice = '.ljust(15) + ' '
result += measure_to_string(dice) + '\n'
result += 'Jaccard = '.ljust(15) + ' '
result += measure_to_string(jaccard) + '\n'

result += '**************************\n'

print result


if not os.path.exists(results_path + results_file):
    with open(results_path + results_file,'w') as f:
        f.write(result)
else:
    with open(results_path + results_file,'a') as f:
        f.write(result)
        
"""


