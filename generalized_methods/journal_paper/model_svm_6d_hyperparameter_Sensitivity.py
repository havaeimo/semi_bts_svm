
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
#import pdb
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

    return {'finaltrainset':finaltrainset, 'testset':testset ,'ground_truth':lbl, 'validset':validset, 'trainset':trainset}    

def compute_error_mean_and_sterror(costs):
    classif_errors = np.hstack(costs)
    classif_mean = classif_errors.mean()
    classif_sterror = classif_errors.std(ddof=1)/np.sqrt(classif_errors.shape[0])

    return classif_mean, classif_sterror

def find_best_model(hyperparams_grid,datasets):
    
    best_val_error = np.inf
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
    return best_hyperparams   

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
    
    dice_mean = train_and_test(svm, datasets)
    end_time = time.clock()
    processing_time = end_time - start_time
    return [dice_mean , processing_time]



def svm_model2(params, datasets):
    from sklearn import svm

    def My_kernel(x,y):
 
        #gamma1 = 1
        #gamma2 = 9  # gamma1 and gamma2 are std^2
        x1 = x[:,0:3]
        y1 = y[:,0:3]
        # constructing the gram matrix        
        d1 = -2 * np.dot(x1,y1.T) + np.tile((x1 * x1).sum(axis=1).reshape(len(x1),1) , (1,len(y1))) + np.tile((y1 * y1).sum(axis=1).reshape(1,len(y1)) , (len(x1),1)) 

        #kernel_1 = 1/(np.sqrt(gamma1*2*3.14))* np.exp( - d1 /(2*gamma1))
        kernel_1 = np.exp(-gamma1 * d1 )
        
        x2 = x[:,3:]
        y2 = y[:,3:]
        d2 = -2 * np.dot(x2,y2.T) + np.tile((x2 * x2).sum(axis=1).reshape(len(x2),1) , (1,len(y2))) + np.tile((y2 * y2).sum(axis=1).reshape(1,len(y2)) , (len(x2),1)) 

        #kernel_2 = 1/(np.sqrt(gamma1*2*3.14))* np.exp(- d2 /(2*gamma2))
        kernel_2 =  np.exp(-gamma2 * d2 )

        return kernel_1 * kernel_2



    start_time = time.clock()
    use_weights = False    
    if use_weights:
        label_weights = finaltrainset.metadata['label_weights']
    else:
        label_weights = None
        
    output_probabilities = True # Or False!
    try:
        # Create SVMClassifier with hyper-parameters
        gamma1 = params[0]
        gamma2 = params[1]
        clf = svm.SVC( shrinking=True, kernel=My_kernel , gamma=0 ,coef0=0,C=params[2],class_weight=label_weights, probability=output_probabilities)
    except Exception as inst:
        print "Error while instantiating SVMClassifier (required hyper-parameters are probably missing)"
        print inst
        sys.exit()
    dice_mean = train_and_test_model_svm_sklearn(clf, datasets)
    end_time = time.clock()
    processing_time = end_time - start_time
    return [dice_mean , processing_time]



def train_and_test(svm,datasets):

    testset = datasets['testset']
    finaltrainset = datasets['finaltrainset']
    svm.train(finaltrainset)
    outputs, costs = svm.test(testset)
    

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
    dice = dice[~np.isnan(dice)]
    return dice.mean()



def train_and_test_model_svm_sklearn(clf,datasets):

    testset = datasets['testset']
    finaltrainset = datasets['finaltrainset']
    X_test = np.array([x for x,y in testset])
    Y_test = np.array([y for x,y in testset])
    X_finaltrain = np.array([x for x,y in finaltrainset])
    Y_finaltrain = np.array([y for x,y in finaltrainset])
    st_time = time.time()
    clf.fit(X_finaltrain, Y_finaltrain)

    print 'Testing...'

    outputs = np.zeros(len(X_test))
    probabilities = np.zeros((len(X_test),len(clf.classes_)))

    minibatch_size = int(len(X_test)/200000)+1;
    #minibatch_size = 5
    chunked_testset =  np.array_split( X_test,minibatch_size)
  
    outputs = np.array([]).reshape(1,-1)
    probabilities = np.array([]).reshape(-1,len(clf.classes_))
     
    for i,test_batch in enumerate(chunked_testset):

        output_batch = clf.predict(test_batch)
        outputs = np.c_[outputs, output_batch.reshape(1,-1)]

        probabilities_batch = clf.predict_proba(test_batch)
        probabilities = np.r_[probabilities, probabilities_batch.reshape(-1, len(clf.classes_))]

    ed_time = time.time()
    print 'timertookd='+ str(ed_time - st_time)
    outputs = outputs[0]
    
    id_to_class = {}
    for label, id in testset.class_to_id.iteritems():
        id_to_class[id] = label

     # Ground truth
    lbl = datasets['ground_truth']
    auto_lbl = np.array([int(id_to_class[output]) for output in outputs]) # Predicted labels

    len_bg = testset.metadata['len_bg']
    lbl = np.append(lbl, [0]*len_bg)
    auto_lbl = np.append(auto_lbl, [0]*len_bg)

    (dice, jaccard, precision, recall) = compute_statistics.compute_eval_multilabel_metrics(auto_lbl, lbl)
    dice = dice[~np.isnan(dice)]
    print dice.mean()
    return dice.mean()
