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
import ipdb




# Loading functions are in this script for now, should be moved to mlpython/datasets later
def create_files(dir_path, train_filename, test_filename, background_filename, input_size , use_spatial_dim ):
    """
    Creates train/valid/test files from the interaction, allpoints and background files
    """
    import mlpython.misc.io as mlio
    import random
    
    targets = set(['0','1','2','3','4'])
    
    train_file, test_file = [os.path.join(dir_path,filename) for filename in [train_filename, test_filename]]
    
    # Load files into memory
    print "Loading train file..."
    
    all_data_interaction, _ = mlio.libsvm_load(filename=train_file,input_size=input_size)
    class_subset = set(np.array(all_data_interaction)[:,-1])
    
    length_interaction = len(all_data_interaction)
    
    # Load test data, filter background points using background file
    print "Loading test file..."
    allpoints_data, _ = mlio.libsvm_load(filename=test_file,input_size=input_size)
    length_allpoints = len(allpoints_data)
    
    if background_filename != None:
        print "Filtering allpoints with background file"
        background_points = []
        with open(os.path.join(dir_path,background_filename),'r') as bg_file:
            background_points = [line[:-1] for line in bg_file] # Remove newline characters
        
        test_data = [test_point for (bg_point, test_point) in zip(background_points, allpoints_data) if bg_point == '0']

    len_test = len(test_data)
    len_bg = length_allpoints - len_test
    
    len_test = len(test_data)

    # Shuffle data
    # random.shuffle shuffles data in place
    print "Shuffling data..."
    random.seed('1234')
    random.shuffle(all_data_interaction)
    random.shuffle(test_data)

    # Split data into train,valid,test (70% train, 20% valid, 10% test)
    len_train = int(0.78 * length_interaction)
    len_valid = length_interaction - len_train
    len_finaltrain = length_interaction

    train_data = all_data_interaction[:len_train]
    valid_data = all_data_interaction[len_train:]
    finaltrain_data = all_data_interaction
    
    lengths = [len_train, len_valid, len_finaltrain, len_test]
    
    # Write train/valid/test files to disk
    print "Creating train/valid/test files..."
    
    def get_line(data):
        line = ""
        line += data[1]
        
        for id,input_value in enumerate(data[0]):
            if input_value != 0:
                line += ' ' + str(id+1) + ':' + str(input_value)
        line += '\n'
        return line
        
        
    # Get label weights
    label_weights = {}
    data = np.array([data[1] for data in finaltrain_data])
    for target in targets:
        nr_label = (data == target).sum()
        if nr_label != 0:
            label_weights[target] = float(len_finaltrain) / float(nr_label)
        else:
            label_weights[target] = 0
    
    # Normalize weights
    index_to_label = {}
    weights = []
    for index, (label, weight) in enumerate(label_weights.iteritems()):
        index_to_label[index] = label
        weights += [weight]
    weights = np.array(weights)
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    for id, weight in enumerate(weights):
        label_weights[index_to_label[id]] = weight
    
    print type(train_data)
    spatial_data = [[t[0][3:6]] for t in test_data]

    if use_spatial_dim == 0:
        train_data = [[t[0][0:3], t[1]] for t in train_data]
        valid_data = [[t[0][0:3], t[1]] for t in valid_data]
        test_data = [[t[0][0:3], t[1]] for t in test_data]
        finaltrain_data = [[t[0][0:3], t[1]] for t in finaltrain_data]
    

    #ipdb.set_trace()    
    print 'size of data ', np.shape(test_data)    
    train_file, valid_file, finaltrain_file, test_file, spatial_file = [os.path.join(dir_path,ds + '.txt') for ds in ['trainset','validset','finaltrainset','testset','spatialinfo']]
    ipdb.set_trace()
    g = open(spatial_file,'w')
    for i in range(np.shape(spatial_data)[0]):
        data = spatial_data[i][0]
        line = ""
        for d in data:
            line += str(d)+' '
        line +='\n'    
        g.write(line)
    g.close()
            
    with open(train_file,'w') as f:
        for data in train_data:
            f.write(get_line(data))
            
    with open(valid_file,'w') as f:
        for data in valid_data:
            f.write(get_line(data))
    
    with open(finaltrain_file,'w') as f:
        for data in finaltrain_data:
            f.write(get_line(data))
            
    with open(test_file,'w') as f:
        for data in test_data:
            f.write(get_line(data))
            
    with open(os.path.join(dir_path,'metadata.txt'),'w') as f:
        for l in lengths:
            f.write(str(l) + '\n')
        f.write(str(len_bg) + '\n')
        class_subset = list(class_subset)
      
        for c in class_subset:
            f.write(c)
        f.write('\n')
        for label, weight in label_weights.iteritems():
            f.write(label + ':' + str(weight) + '\n')



def load_data(dir_path, input_size=6, targets=set(['0','1','2','3','4']), train_filename=None, test_filename=None, background_filename=None, load_to_memory=True, use_spatial_dim = 1):
    """
    Loads a dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**
    
    * ``'input_size'``
    * ``'targets'``
    * ``'length'``

    """
    import mlpython.misc.io as mlio

    # Known metadata
    dir_path = os.path.expanduser(dir_path)

    # Look if the train/valid/test files already exist, if not, load the data and create the files
    train_file, valid_file, finaltrain_file, test_file = [os.path.join(dir_path, ds + '.txt') for ds in ['trainset','validset','finaltrainset','testset']]
    
    if os.path.exists(train_file):
        print "Train/valid/test files exist, loading data..."
    else:
        print "Train/valid/test file do not exist, creating them..."
        if train_filename is None or test_filename is None:
            print 'ERROR, NO TRAIN/TEST FILENAMES GIVEN'
            sys.exit(1)
        else:
            create_files(dir_path,train_filename,test_filename,background_filename,input_size , use_spatial_dim)
        
    # train/valid/test files should exist by now
    if load_to_memory:
        train_data, valid_data, finaltrain_data, test_data = [mlio.libsvm_load(filename=f, input_size=input_size)[0] for f in [train_file, valid_file, finaltrain_file, test_file]]
    else:
        def load_line(line):
            return mlio.libsvm_load_line(line,input_size=input_size)
            
        train_data, valid_data, finaltrain_data, test_data = [mlio.load_from_file(filename=f,load_line=load_line) for f in [train_file, valid_file, finaltrain_file, test_file]]
            

    # Get metadata
    with open(os.path.join(dir_path,'metadata.txt'),'r') as f:
        train_meta,valid_meta,finaltrain_meta,test_meta = [{'input_size':input_size,'length':int(f.readline()[:-1]),'targets':targets} for i in range(4)]
        test_meta['len_bg'] = int(f.readline()[:-1])

        label_weights = {}
        class_subset = f.readline()
        class_subset = class_subset[:-1]
        class_subset = set(class_subset)
        
        for _ in range(len(targets)):
            label, weight = f.readline()[:-1].split(':')
            label_weights[label] = float(weight)
        finaltrain_meta['label_weights'] = label_weights
        train_meta['subset'] = class_subset
        valid_meta['subset'] = class_subset
        finaltrain_meta['subset'] = class_subset
        test_meta['subset'] = class_subset

        
    return {'train':(train_data,train_meta),'valid':(valid_data,valid_meta), 'finaltrain':(finaltrain_data,finaltrain_meta),'test':(test_data,test_meta)}

