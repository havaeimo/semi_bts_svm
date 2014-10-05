import numpy as np
import sys
import os
import pdb


def load_mean_results(directory,model_name):
    with open(os.path.join(directory,model_name + '_results/' + model_name + '_measures.txt' ),'r') as f:
        f.readline() # Header
        all_values = []
        
        while True:
            measures = np.zeros((4,4)) # Precision / recall / dice / jaccard
            
            eof = not f.readline()
            if eof:
                break
            
            for i in range(14):
                f.readline() # Useless info
                
            for measure in measures:
                raw_line = f.readline()
                
                info_line = raw_line[17:-1] # Remove chars before first value + newline
                info_line = info_line.replace('   ',' ') # Remove duplicate spaces
                info_line = info_line.replace('  ',' ') # Remove duplicate spaces
                if info_line[0] == ' ':
                    info_line = info_line[1:]
                
                measure[:] = np.array(info_line.split(' ')).astype(float)
            
            all_values += [measures]    
            
            f.readline() # Garbage
            
    all_values = np.array(all_values)
    result = all_values.mean(0)
    
    return result
        
def load_ranking_results(directory, model_names):
    results = {}
    for model_name in model_names:
        with open(os.path.join(directory, model_name + '_results/' + model_name + '_measures.txt'), 'r') as f:
            f.readline() # Remove header
            
            while True:
                line = f.readline()
                if not line:
                    break
                
                brain_name = line[:-1].split(' : ')[1]
                
                if brain_name not in results:
                    results[brain_name] = {}
                
                for i in range(14):
                    f.readline() # Useless info
                    
                measures = np.zeros((4,4)) # Precision / recall / dice / jaccard
                
                for measure in measures:
                    raw_line = f.readline()
                
                    info_line = raw_line[17:-1] # Remove chars before first value + newline
                    info_line = info_line.replace('   ',' ') # Remove duplicate spaces
                    info_line = info_line.replace('  ',' ') # Remove duplicate spaces
                    if info_line[0] == ' ':
                        info_line = info_line[1:]

                    measure[:] = np.array(info_line.split(' ')).astype(float)
                
                f.readline() # Garbage
                method = model_name
                
                # Some measures are 'nan'...
                results[brain_name][method] = np.ma.masked_array(measures,not np.isnan)
    
    # First, rank methods for all measures (16)
    brain_measures_ranking = {}
    
    # Then get a global rank for each method for the current brain
    brain_ranking = {}
    
    for brain_name in results:
        brain_ranking[brain_name] = {}
        brain_measures_ranking[brain_name] = {}
        
        # For all measures
        for i in range(4):
            for j in range(4):
                # Sort methods on this particular measure
                sorted_methods = sorted(results[brain_name],key=lambda x: results[brain_name][x][i][j],reverse=True)
                
                # Then rank methods
                for rank, method in enumerate(sorted_methods,1):
                    if method not in brain_measures_ranking[brain_name]:
                        brain_measures_ranking[brain_name][method] = np.zeros((4,4))
                    
                    brain_measures_ranking[brain_name][method][i][j] = rank

        # Get mean rank for each method, then rank on this value
        for method in brain_measures_ranking[brain_name]:
            brain_ranking[brain_name][method] = brain_measures_ranking[brain_name][method].mean()
        
            
    return brain_ranking, brain_measures_ranking
        
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print 'Usage: python 01_get_ranking_and_mean_measures.py directory [model_names]'
        sys.exit(1)
    elif len(sys.argv) == 2:
        directory = sys.argv[1]
        # Default model names
        model_names = ['sklearn_knn','libsvm']
    else:
        directory = sys.argv[1]
        model_names = sys.argv[2:]
        
        
    # Mean results
    result = ''
    for model_name in model_names:
        result += 'Model name : ' + model_name + '\n'
        result += str(load_mean_results(directory, model_name)) + '\n\n'
    
    mean_results_filename = 'mean_measures.txt'
    with open(directory + mean_results_filename, 'w') as f:
        f.write(result)
            
    print result
            
    # Detailed ranking results
    brain_ranking, brain_measures_ranking = load_ranking_results(directory,model_names)
    
    result = ''
    for brain_name in brain_measures_ranking:
        result += brain_name + '\n'
        for method in brain_measures_ranking[brain_name]:
            result += method + ':\n'
            result += str(brain_measures_ranking[brain_name][method])
            result += '\n'
        result += '\n'
        
    with open(directory + 'ranking_01_detailed_results.txt','w') as f:
        f.write(result)
        
    # Summarized ranking results
    result = ''
    for brain_name in brain_ranking:
        result += brain_name + '\n'
        for method in brain_ranking[brain_name]:
            result += method + ':' + str(brain_ranking[brain_name][method]) + '\n'
        result += '\n'
        
    ranking_results_filename = directory + 'ranking_02_summarized_results.txt'
    with open(ranking_results_filename, 'w') as f:
        f.write(result)
        
        
