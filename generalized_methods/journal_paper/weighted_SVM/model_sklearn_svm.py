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
import pdb
#import ipdb ipdb
def load_data(dataset_directory , dataset_name):
    print "Loading datasets ..."
    import os
    repo = os.environ.get('MLPYTHON_DATASET_REPO')
    if repo is None:
        raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
    dataset_dir = os.path.join(os.environ.get('MLPYTHON_DATASET_REPO') + '/' + dataset_directory, dataset_name)    
    
    input_size = 6 
    spatial_dimensions = 1
    all_data = data_utils.load_data(dir_path=dataset_dir, input_size=input_size, train_filename=None, test_filename=None, background_filename=None,load_to_memory=False)

    train_data, train_metadata = all_data['train']
    valid_data, valid_metadata = all_data['valid']
    finaltrain_data, finaltrain_metadata = all_data['finaltrain']
    test_data, test_metadata = all_data['test']
    lbl = np.array([int(data[1]) for data in test_data])

    def reduce_dimensionality(mlproblem_data, mlproblem_metadata):
        mlproblem_metadata['input_size'] = 3  # we need to change the input size from 6 to 3. 
        return [mlproblem_data[0][:3] , mlproblem_data[1]]

    if spatial_dimensions ==1:      
        import mlpython.mlproblems.classification as mlpb
        trainset = mlpb.ClassificationProblem(train_data, train_metadata)
        validset = trainset.apply_on(valid_data,valid_metadata)
        finaltrainset = trainset.apply_on(finaltrain_data,finaltrain_metadata)
        testset = trainset.apply_on(test_data,test_metadata)

    elif spatial_dimensions ==0:
        import mlpython.mlproblems.generic as mlpg
        trainset = mlpg.PreprocessedProblem(data = train_data , metadata = train_metadata , preprocess = reduce_dimensionality)
        validset = trainset.apply_on(valid_data, valid_metadata)
        testset = trainset.apply_on(test_data, test_metadata)
        finaltrainset = trainset.apply_on(finaltrain_data, finaltrain_metadata)
        import mlpython.mlproblems.classification as mlpb
        trainset = mlpb.ClassificationProblem(trainset, trainset.metadata)
        validset = trainset.apply_on(validset,validset.metadata)
        finaltrainset = trainset.apply_on(finaltrainset,finaltrainset.metadata)
        testset = trainset.apply_on(testset,testset.metadata)

    return {'finaltrainset':finaltrainset, 'testset':testset ,'ground_truth':lbl}  


    
def Cost(outputs,targets):
    """
    Computes and returns the outputs of the Learner as well as the errors of 
    those outputs for ``dataset``:
    - the errors should be a Numpy 2D array of size
      len(dataset) by 2
    - the ith row of the array contains the errors for the ith example
    - the errors for each example should contain 
      the 0/1 classification error (first element) and the 
      regularized negative log-likelihood (second element)
     """
    errors = np.zeros((len(targets),1))
    
    t=0
    for target in targets:
        output = outputs[t]
        errors[t] = output!= target
        t+=1
        
    return errors

def compute_error_mean_and_sterror(costs):
    classif_errors = np.hstack(costs)
    classif_mean = classif_errors.mean()
    classif_sterror = classif_errors.std(ddof=1)/np.sqrt(classif_errors.shape[0])

    return classif_mean, classif_sterror


def My_kernel(x,y):
 
    sigma_1 = 1
    sigma_2 = 9  # sigma_1 and sigma_2 are std^2

    x1 = x[:,0:3]
    y1 = y[:,0:3]
    
    d1 = -2 * np.dot(x1,y1.T) + np.tile((x1 * x1).sum(axis=1).reshape(len(x1),1) , (1,len(y1))) + np.tile((y1 * y1).sum(axis=1).reshape(1,len(y1)) , (len(x1),1)) 


    #kernel_1 = 1/(np.sqrt(sigma_1*2*3.14))* np.exp( - d1 /(2*sigma_1))
    kernel_1 = np.exp(-sigma)
    x2 = x[:,3:]
    y2 = y[:,3:]
    d2 = -2 * np.dot(x2,y2.T) + np.tile((x2 * x2).sum(axis=1).reshape(len(x2),1) , (1,len(y2))) + np.tile((y2 * y2).sum(axis=1).reshape(1,len(y2)) , (len(x2),1)) 

    kernel_2 = 1/(np.sqrt(sigma_1*2*3.14))* np.exp(- d2 /(2*sigma_2))
    
    return kernel_1 * kernel_2




print "Setting hyperparameters gridsearch..."
best_hyperparams = None
best_val_error = np.inf

   
# defining arrays of train , valid test for sklearn svm to use (takes mlproble and returns numpy array)
X_train = np.array([x for x,y in trainset])
Y_train = np.array([y for x,y in trainset])
X_valid = np.array([x for x,y in validset])
Y_valid = np.array([y for x,y in validset])
X_finaltrain = np.array([x for x,y in finaltrainset])
Y_finaltrain = np.array([y for x,y in finaltrainset])
X_test = np.array([x for x,y in testset])
Y_test = np.array([y for x,y in testset])


if use_weights:
    label_weights = finaltrainset.metadata['label_weights']
else:
    label_weights = None
    
output_probabilities = True # Or False!


print "Pretraining..."
for params in hyperparams_grid:
    try:
        # Create SVMClassifier with hyper-parameters
        clf = svm.SVC( shrinking=True, kernel=My_kernel ,degree=params[1],gamma=params[2],coef0=params[3],C=params[4],class_weight=label_weights, probability=output_probabilities)
        
    except Exception as inst:
        print "Error while instantiating SVMClassifier (required hyper-parameters are probably missing)"
        print inst
        sys.exit()
        
    #ipdb.set_trace()      
    clf.fit(X_train,Y_train)
  
    outputs = clf.predict(X_valid)
 
    costs = Cost(outputs, Y_valid)

     
    
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
#gamma = 5.0
#C = 1
#best_hyperparams = ['rbf',3,gamma,0,C]
best_clf = svm.SVC(shrinking=True, kernel=My_kernel ,degree=best_hyperparams[1],gamma=best_hyperparams[2],coef0=best_hyperparams[3],C=best_hyperparams[4],class_weight=label_weights, probability=output_probabilities)
best_clf.fit(X_finaltrain, Y_finaltrain)

print 'Testing...'

outputs = np.zeros(len(X_test))
probabilities = np.zeros((len(X_test),len(clf.classes_)))

minibatch_size = int(len(X_test)/10000)+1;
chunked_testset =  np.array_split( X_test,minibatch_size)
#ipdb.set_trace()
outputs = np.array([]).reshape(1,-1)
probabilities = np.array([]).reshape(-1,len(clf.classes_))

for i,test_batch in enumerate(chunked_testset):
    pdb.set_trace()
    output_batch = best_clf.predict(test_batch)
    outputs = np.c_[outputs, output_batch.reshape(1,-1)]

    probabilities_batch = best_clf.predict_proba(test_batch)
    probabilities = np.r_[probabilities, probabilities_batch.reshape(-1, len(clf.classes_))]



outputs = outputs[0]
costs = Cost(outputs,Y_test)
errors = compute_error_mean_and_sterror(costs)
#error = errors[0]

print
print 'Classification error on test set : ' + str(error)
print "****************************************"


# Evaluation (compute_statistics.py)
id_to_class = {}
for label, id in testset.class_to_id.iteritems():
    id_to_class[id] = label
    
lbl = np.array([int(data[1]) for data in test_data]) # Ground truth
auto_lbl = np.array([int(id_to_class[output]) for output in outputs]) # Predicted labels
"""
WARNING!
If using probability distributions, we need to remap the ids to the right class (for all outputs)
"""
temp = np.c_[outputs,probabilities,np.zeros((len(outputs),1))]
outputs = temp

# Write outputs to disk if result file does not already exist
results_path = output_folder + 'libsvm_results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)
if use_weights:
    output_file = results_path + dataset_name + '_weightedlabels_output_txt'
else:
    output_file = results_path + dataset_name + '_libsvm_output.txt'
if not os.path.exists(output_file):
    with open(output_file,'w') as f:
        for output, data in zip(outputs, test_data):
            line = ""
            for l in output:
                line += str(l) + ' '

            line +=  str(data[0][3]) + ' '
            line +=  str(data[0][4]) + ' '
            line +=  str(data[0][5]) + '\n'
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
        



