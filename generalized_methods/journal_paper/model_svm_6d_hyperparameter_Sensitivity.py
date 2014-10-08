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

def compute_error_mean_and_sterror(costs):
    classif_errors = np.hstack(costs)
    classif_mean = classif_errors.mean()
    classif_sterror = classif_errors.std(ddof=1)/np.sqrt(classif_errors.shape[0])

    return classif_mean, classif_sterror

def svm_model(dataset_directory, dataset_name, params, datasets):
    start_time = time.clock()
    use_weights = False    
    if use_weights:
        label_weights = finaltrainset.metadata['label_weights']
    else:
        label_weights = None
        
    output_probabilities = True # Or False!
    try:
        # Create SVMClassifier with hyper-parameters
        svm = SVMClassifier(shrinking=True, kernel=params[0],degree=params[1],gamma=params[2],coef0=params[3],C=params[4],label_weights=label_weights, output_probabilities=output_probabilities)
    except Exception as inst:
        print "Error while instantiating SVMClassifier (required hyper-parameters are probably missing)"
        print inst
        sys.exit()
    finaltrainset = datasets['finaltrainset']
    svm.train(finaltrainset)
    testset = datasets['testset']
    outputs, costs = svm.test(testset)

    end_time = time.clock()
    processing_time = end_time - start_time

    id_to_class = {}
    for label, id in testset.class_to_id.iteritems():
        id_to_class[id] = label
        
     # Ground truth
    lbl = datasets['ground_truth'] 
    auto_lbl = np.array([int(id_to_class[output[0]]) for output in outputs]) # Predicted labels

    len_bg = testset.metadata['len_bg']
    lbl = np.append(lbl, [0]*len_bg)
    auto_lbl = np.append(auto_lbl, [0]*len_bg)
    
    (dice, jaccard, precision, recall) = compute_statistics.compute_eval_multilabel_metrics(auto_lbl, lbl)
    return [dice.mean(), processing_time]
    
