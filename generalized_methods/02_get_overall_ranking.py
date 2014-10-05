import numpy as np
import sys
import os
import pdb




if __name__ == '__main__':
    if len(sys.argv) == 2:
        directory = sys.argv[1]
        filename = 'ranking_02_summarized_results.txt'
    else:
        print 'Usage : python 02_get_overall_ranking.py directory'
        print 'Example : python 02_get_overall_ranking.py results_2013_12_05/'
        sys.exit(0)
    
    with open(directory + filename,'r') as f:
        line = f.readline() # Brain name
        total = 0
        rank_count = {}
        rank_sum = {}
        
        while line:
            line = f.readline() # First result of brain
            while line != '\n' and line:
                tokens = line[:-1].split(':')
                
                measure_name = tokens[0]
                rank = float(tokens[1])
                
                if measure_name not in rank_count:
                    rank_count[measure_name] = 1
                    rank_sum[measure_name] = rank
                else:
                    rank_count[measure_name] += 1
                    rank_sum[measure_name] += rank

                line = f.readline() # Next result, might be '\n'
            line = f.readline() # Brain name


        with open(directory + 'ranking_03_overall.txt', 'w') as f:
            for name in rank_sum:
                f.write(name + ' : ')
                mean = rank_sum[name] / float(rank_count[name])
                f.write(str(mean) + '\n')
