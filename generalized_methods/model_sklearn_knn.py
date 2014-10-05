import numpy as np
import os
import sys
import fcntl
import copy
from string import Template
import mlpython.datasets.store as dataset_store
import compute_statistics
import time
import data_utils
import pdb



sys.argv.pop(0);	# Remove first argument

# Check if every option(s) from parent's script are here.

arg_length = len(sys.argv)
if arg_length not in [3,5,6]:
    print len(sys.argv)
    print "Usage: python model_sklearn_knn.py mlpython_dataset_directory dataset_name output_folder [train_filename test_filename [background_filename]]"
    print "Filenames for the train/test/background files are necessary on the first use, they are not needed later on, as trainset/validset/testset files are created"
    print "Ex.: python model_sklearn_knn.py brains_data/Brats-2_training/ HG_0001 . interaction.txt allpoints.txt background.txt"
    sys.exit()


# Get arguments
dataset_directory = sys.argv[0]
dataset_name = sys.argv[1]
output_folder = sys.argv[2]
train_filename = None
test_filename = None
background_filename = None

if arg_length >= 4:
    train_filename = sys.argv[3]
    test_filename = sys.argv[4]
    
    if arg_length == 6:
        background_filename = sys.argv[5]


input_size = 6 # Do we want this as command-line parameter?


# Set the constructor
str_ParamOption = ""
str_ParamOptionValue = ""

# Load data

start_time = time.clock()


dataset_dir = None
if dataset_dir is None:
    # Try to find dataset in MLPYTHON_DATASET_REPO
    import os
    repo = os.environ.get('MLPYTHON_DATASET_REPO')
    if repo is None:
        raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
    dataset_dir = os.path.join(os.environ.get('MLPYTHON_DATASET_REPO') + '/' + dataset_directory, dataset_name)

all_data = data_utils.load_data(dir_path=dataset_dir, input_size=input_size, train_filename=train_filename, test_filename=test_filename, background_filename=background_filename,load_to_memory=False)

train_data, train_metadata = all_data['train']
valid_data, valid_metadata = all_data['valid']
finaltrain_data, finaltrain_metadata = all_data['finaltrain']
test_data, test_metadata = all_data['test']


def compute_error_mean_and_sterror(costs):
    classif_errors = np.hstack(costs)
    classif_mean = classif_errors.mean()
    classif_sterror = classif_errors.std(ddof=1)/np.sqrt(classif_errors.shape[0])

    return classif_mean, classif_sterror


from sklearn.neighbors import KNeighborsClassifier

print "Setting hyperparameters gridsearch..."
best_hyperparams = None
best_val_error = np.inf

k = [1,2,3,4,5,10,15,20]

print "Pretraining..."
#X,Y = zip(*[[data[0],int(data[1])] for data in train_data]) # X: Train points, Y: Targets
X,Y = [data[0] for data in train_data],[int(data[1]) for data in train_data]

#Z, targets = zip(*[[data[0],int(data[1])] for data in valid_data]) # Z: Points to predict
Z, targets = [data[0] for data in valid_data],[int(data[1]) for data in valid_data]

for params in k:
    #print
    #print "Hyperparams : K=" + str(params)
    try:
        # Create KNeighborsClassifier with hyper-parameters
        knn = KNeighborsClassifier(params)
    except Exception as inst:
        print "Error while instantiating KNeighborsClassifier (required hyper-parameters are probably missing)"
        print inst
        sys.exit()

    knn.fit(X,Y)
    outputs = knn.predict(Z)
    costs = np.ones((len(outputs),1))
    for target,pred,cost in zip(targets,outputs,costs):
        if target == pred:
            cost[0] = 0
    
    errors = compute_error_mean_and_sterror(costs)
    error = errors[0]
    
    #print "Error : " + str(errors)
    
    if error < best_val_error:
        best_val_error = error
        best_hyperparams = params


print
print 'Classification error on valid set : ' + str(best_val_error)
print
print "Training..."
# Train KNN

#X,Y = zip(*[[data[0],int(data[1])] for data in finaltrain_data])
X,Y = [data[0] for data in finaltrain_data],[int(data[1]) for data in finaltrain_data]

K = best_hyperparams
knn = KNeighborsClassifier(K)
knn.fit(X, Y)

print 'Testing...'
#Z, lbl = zip(*[[data[0],int(data[1])] for data in test_data])
Z, lbl = [data[0] for data in test_data],[int(data[1]) for data in test_data]

outputs = knn.predict(Z)
costs = np.ones((len(outputs),1))
for target,pred,cost in zip(lbl,outputs,costs):
    if target == pred:
        cost[0] = 0
errors = compute_error_mean_and_sterror(costs)
error = errors[0]
print 'Classification error on test set : ' + str(error)




print "****************************************"


# Evaluation (compute_statistics.py)
    
#lbl = np.array([int(data[1]) for data in test_data]) # Ground truth
lbl = np.array(lbl)
#auto_lbl = np.array([int(id_to_class[output[0]]) for output in outputs]) # Predicted labels
auto_lbl = outputs

# Write outputs to disk if result file does not already exist
results_path = output_folder + 'sklearn_knn_results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)
output_file = results_path + dataset_name + '_sklearn_knn_output.txt'
if not os.path.exists(output_file):
    with open(output_file,'w') as f:
        for label, data in zip(auto_lbl, test_data):
            line = str(label) + ' '
            line += '1:' + str(data[0][3]) + ' '
            line += '2:' + str(data[0][4]) + ' '
            line += '3:' + str(data[0][5]) + '\n'
            f.write(line)

# Compute processing time
end_time = time.clock()
print "Processing time : " + str(float(end_time - start_time) / 60) + ' minutes'
print

def string_debug(labels):
    result = ''
    for i in range(5):
        result += "# data with label " + str(i) + " : " + str( (labels == i).sum() ).rjust(8) + '\n'
    return result
        
def hyper_to_string(hyperparams):
    result = ''
    result += 'K = ' + str(hyperparams)
    return result
        
def measure_to_string(measure):
    result = ''
    for value in measure:
        result += str(value)[:5].rjust(6)
    return result


result = ''

# Fill prediction / ground truth with zeros for all background points
"""
len_bg = testset.metadata['len_bg']
lbl = np.append(lbl, [0]*len_bg)
auto_lbl = np.append(auto_lbl, [0]*len_bg)
"""

(dice, jaccard, precision, recall) = compute_statistics.compute_eval_multilabel_metrics(auto_lbl, lbl)

lbl_result = string_debug(lbl)
auto_lbl_result = string_debug(auto_lbl)

results_file = 'sklearn_knn_measures.txt'
if not os.path.exists(results_path + results_file):
    result += 'Results = [Edema, non-enhanced tumor, enhanced tumor, complete (abnormality vs healthy)]\n'
    
result += 'Dataset : ' + dataset_name + '\n'
result += 'Model : KNN\n'

result += 'Best hyperparameters : ' + hyper_to_string(K) + '\n'

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
