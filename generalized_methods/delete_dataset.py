import os

"""
This is a script to delete some files (the ones listed in the file_list list). 
"""
root_path = '/home/local/USHERBROOKE/havm2701/ml_datasets/BRATS2013/Brats-2_training/'
directory_list = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path,f))]
file_list = ['trainset','validset', 'testset','metadata','finaltrainset']
#file_list = ['metadata']
for l in directory_list:
	for i in  file_list:
		file_path = root_path + l +'/'+ i + '.txt' 
		if os.path.isfile(file_path):
			print file_path
			os.remove(file_path)
