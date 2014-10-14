import numpy as np
import os
import sys
import fcntl
import copy
sys.path.append('/home/local/USHERBROOKE/havm2701/git.repos/semi_bts_svm/semi_bts_svm/generalized_methods/')
from string import Template
import mlpython.datasets.store as dataset_store
from mlpython.learners.third_party.libsvm.classification import SVMClassifier
import compute_statistics
import time
import data_utils
#import ipdb




#test_data, test_metadata = all_data['test']
#final_data = data_utils.data_reduction(finaltrain_data , factor = 5)
def create_datasets(all_data):
    train_data, train_metadata = all_data['train']
    valid_data, valid_metadata = all_data['valid']
    finaltrain_data, finaltrain_metadata = all_data['finaltrain']
    test_data, test_metadata = all_data['test']
    lbl = np.array([int(data[1]) for data in test_data])
    spatial_dimensions = 1

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

    return {'trainset':trainset,'validset':validset ,'finaltrainset':finaltrainset, 'testset':testset ,'ground_truth':lbl}


def compute_error_mean_and_sterror(costs):
    classif_errors = np.hstack(costs)
    classif_mean = classif_errors.mean()
    classif_sterror = classif_errors.std(ddof=1)/np.sqrt(classif_errors.shape[0])

    return classif_mean, classif_sterror

def svm_model(datasets):
    print "Setting hyperparameters gridsearch..."
    best_hyperparams = None
    best_val_error = np.inf

    finaltrainset = datasets['finaltrainset']
    trainset = datasets['trainset']
    validset = datasets['validset']
    testset = datasets['testset']
    lbl = datasets['ground_truth']

    output_probabilities = True # Or False!
    kernels = ['rbf','sigmoid']
    #degrees = [1,2,3,4,5,7,10,15]
    gammas = [0.01,0.1,1,5,10,50,100,200,500,1000]
    #coef0s = [-10,-1,-0.1,-0.01,0,0.001,0.01,0.1,1,2,5,10,20]
    Cs = [1,5,10,25,50,75,100,200,500,1000,1500]

    hyperparams_grid = []
    # Rbf kernel parameters
    start_time = time.clock()
    for gamma in gammas:
        for C in Cs:
            hyperparams_grid.append(['rbf', 3, gamma, 0, C])
    
    use_weights = None               
    if use_weights:
        label_weights = finaltrainset.metadata['label_weights']
    else:
        label_weights = None
        
    output_probabilities = False # Or False!

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
    #gamma = 5.0
    #C = 1
    #best_hyperparams = ['rbf',3,gamma,0,C]
    best_svm = SVMClassifier(shrinking=True, kernel=best_hyperparams[0],degree=best_hyperparams[1],gamma=best_hyperparams[2],coef0=best_hyperparams[3],C=best_hyperparams[4],label_weights=label_weights, output_probabilities=output_probabilities)
    best_svm.train(finaltrainset)

    print 'Testing...'
    outputs, costs = best_svm.test(testset)
    end_time = time.clock()
    processing_time = end_time - start_time
    
    errors = compute_error_mean_and_sterror(costs)
    error = errors[0]

    print
    print 'Classification error on test set : ' + str(error)
    print "****************************************"


    # Evaluation (compute_statistics.py)
    id_to_class = {}
    for label, id in testset.class_to_id.iteritems():
        id_to_class[id] = label
        

    auto_lbl = np.array([int(id_to_class[output[0]]) for output in outputs]) # Predicted labels

    len_bg = testset.metadata['len_bg']
    lbl = np.append(lbl, [0]*len_bg)
    auto_lbl = np.append(auto_lbl, [0]*len_bg)
    (dice, jaccard, precision, recall) = compute_statistics.compute_eval_multilabel_metrics(auto_lbl, lbl)
    dice = dice[~np.isnan(dice)]
    return [dice.mean(), processing_time]


