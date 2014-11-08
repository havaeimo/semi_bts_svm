import numpy as np
import os
import sys
import fcntl
import copy
#import pdb
from multiprocessing import Pool

sys.path.append('/home/local/USHERBROOKE/havm2701/git.repos/semi_bts_svm/semi_bts_svm/generalized_methods/')

from string import Template
import mlpython.datasets.store as dataset_store
from mlpython.learners.third_party.libsvm.classification import SVMClassifier
import compute_statistics
import time
import data_utils
import mlpython.mlproblems.generic as mlpg
from mlpython.mlproblems.generic import SubsetProblem
# make a dictionary of the selected brains with their best c and gamma parameters. 
#brain_list = {'HG_0002': [1,50], 'HG_0001': [1,0.01], 'HG_0003': [1,1], 'HG_0010': [1,50], 'HG_0008': [1,5], 'HG_0012': [50,50], 'HG_0011': [1,5], 'HG_0022': [1,5], 'HG_0025': [1,50], 'HG_0027': [1,10], 'LG_0008': [1,200], 'LG_0001': [1,50], 'LG_0006': [1,1], 'LG_0015': [100,200], 'HG_0014': [1,5]}


  

def compute_error_mean_and_sterror(costs):
    classif_errors = np.hstack(costs)
    classif_mean = classif_errors.mean()
    classif_sterror = classif_errors.std(ddof=1)/np.sqrt(classif_errors.shape[0])

    return classif_mean, classif_sterror

def find_best_model(hyperparams_grid):
    
    best_val_error = np.inf
    best_hyperparams = None
    validset = datasets['validset']
    trainset = datasets['trainset']
    output_probabilities = True 
    label_weights = None
    
    
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
    return (best_hyperparams ,best_val_error)  

def svm_model(dataset_directory, dataset_name, params, datasets):
    start_time = time.clock()
    use_weights = False
    output_probabilities = True 
    label_weights = None    
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
    #finaltrainset = datasets['finaltrainset']
    #svm.train(finaltrainset)
    #testset = datasets['testset']
    #outputs, costs = svm.test(testset)
    dice_mean = train_and_test(svm, datasets)
    end_time = time.clock()
    processing_time = end_time - start_time
    return [dice_mean , processing_time]



def svm_train_test(input_arg):


    label_weights = None
    best = input_arg[1]
    testset = input_arg[0]    
    output_probabilities = True # Or False!
    try:
        # Create SVMClassifier with hyper-parameters
        svm_best = SVMClassifier(shrinking=True, kernel=best[0],degree=best[1],gamma=best[2],coef0=best[3],C=best[4],label_weights=label_weights, output_probabilities=output_probabilities)
    except Exception as inst:
        print "Error while instantiating SVMClassifier (required hyper-parameters are probably missing)"
        print inst
        sys.exit()
   

    finaltrainset = datasets['finaltrainset']
    svm_best.train(finaltrainset)
    outputs, costs = svm_best.test(testset)
    return (outputs,costs)
    



def svm_test(testset):

    outputs = svm.test(testset)
   
    id_to_class = {}
    for label, id in testset.class_to_id.iteritems():
        id_to_class[id] = label
        
     # Ground truth
    auto_lbl = np.array([int(id_to_class[output[0]]) for output in outputs]) # Predicted labels
    return auto_lbl


if __name__ == '__main__':
    sys.argv.pop(0); # Remove first argument

    # Get arguments
    dataset_directory = sys.argv[0]
    dataset_name = sys.argv[1]
    output_folder = sys.argv[2]


dataset_dir = None
if dataset_dir is None:
    # Try to find dataset in MLPYTHON_DATASET_REPO
    import os
    repo = os.environ.get('MLPYTHON_DATASET_REPO')
    if repo is None:
        raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
    dataset_dir = os.path.join(os.environ.get('MLPYTHON_DATASET_REPO') + '/' + dataset_directory, dataset_name)

    gammas = [0.01,1,5,10,50,100]
    Cs = [1,10,50,100]

    hyperparams_grid = []
    all_data = data_utils.load_data(dataset_dir)
    test = all_data['test']
    fulltrain_backup = all_data['finaltrain']    
    factor = 500
    all_data = data_utils.data_reduction(fulltrain_backup , factor)
    all_data['test'] = test
    print len(fulltrain_backup[0])
    print len(all_data['finaltrain'][0])
    train = [int(t[1]) for t in all_data['train'][0]] 
    valid = [int(v[1]) for v in all_data['valid'][0]]
    finaltrain = [int(f[1]) for f in all_data['finaltrain'][0]]
    train = np.asarray(train)
    valid = np.asarray(valid)
    finaltrain = np.asarray(finaltrain)
    datasets = data_utils.create_datasets(all_data)
    testset = datasets['testset']




    resultg1, resultg2 = '' ,''
    for gamma in gammas:
        for C in Cs:
            hyperparams_grid.append(['rbf', 3, gamma, 0, C])

    cores = 8
    pool = Pool(processes=cores)
    len_process = np.int(np.ceil(len(hyperparams_grid)/(cores*1.0)))
    core_params = []
    for index in range(cores):
    	core_params.append(hyperparams_grid[index*len_process:index*len_process+len_process])

    
    multiprocessing_start_time = time.time()

    print 'finidng hyper-parameters...'
    results = pool.map(find_best_model, (core_params[0],core_params[1],core_params[2],core_params[3],core_params[4],core_params[5],core_params[6],core_params[7]))
    er = [t[1] for t in results]
    er = np.array(er)
    index_best = er.argmin()
    best_params = results[index_best][0]

    multiprocessing_end_time = time.time()

    multiprocessing_processing_time = multiprocessing_end_time - multiprocessing_start_time
    print 'finidng hyper-parameters took ' +str(multiprocessing_processing_time)+' seconds'
    '''
    singlethread_start_time = time.clock()
    best_params = find_best_model(hyperparams_grid)
    singlethread_end_time = time.clock()
    singlethread_processing_time = singlethread_end_time - singlethread_start_time
    #dice_g , processed_timeg = svm_model(dataset_directory, brain, params, datasets)
    '''
    print 'training...'
    
    len_testset_core = np.int(np.ceil(len(testset)/(cores*1.0)))

    core_testset = []
    for index in range(cores):
        core_testset.append([[t for t in SubsetProblem(testset,subset=set(range(index*len_testset_core,index*len_testset_core+len_testset_core)))],best_params])
    print 'testing...'
    
    t1 = time.time()	
    pool2 = Pool(processes=8)
    predictions = pool2.map(svm_train_test, (core_testset[0],core_testset[1],core_testset[2],core_testset[3],core_testset[4],core_testset[5],core_testset[6],core_testset[7]))
    t2 = time.time()
    print 'testing took ' + str((t2-t1)) 
    #core_testset = parrallelize_testset(testset,cores)
    '''
    t1 = time.time()
    predictions = svm_train_test([testset,best_params])

    t2 = time.time()
    print 'testing took ' + str((t2-t1))   
    ''' 
