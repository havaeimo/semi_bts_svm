#!/bin/bash
# Usage: python svm_brains_test.py brain_name use_label_weights
# Usage: python knn_brains.py brain_name

# 13/11/2013, 14:42
#launch_jobs brains_svm_no_weights localhost:4 'python svm_brains_test.py HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22,24} False results_svm_no_weights.txt'

# 13/11/2013, 17:29
#launch_jobs brains_svm_with_weights localhost:4 'python svm_brains_test.py HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22,24} True results_svm_with_weights.txt'

# 13/11/2013 17:55
#launch_jobs brains_knn localhost:3 'python knn_brains.py HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22,24} results_knn.txt'

# 04/12/2013, 13:45
#launch_jobs console_output_libsvm localhost:4 'python svm_brains_test.py HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22}'

# 04/12/2013, 15:00
#launch_jobs console_output_knn localhost:4 'python knn_brains.py HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22}'

# 05/12/2013, 10:53
#launch_jobs console_output_sklearn_decisiontree localhost:4 'python sklearn_decisiontree.py HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22}'

# 11/12/2013, 18:30
#launch_jobs console_output_sklearn_svm localhost:4 'python sklearn_svm.py HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22}'

# 21/12/2013, 09:30
#launch_jobs console_output_sklearn_lda localhost:2 'python sklearn_lda.py HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22}'

# 21/12/2013, 14:40
#launch_jobs 20131213_console_output_sklearn_decisiontree localhost:1 'python sklearn_decisiontree.py HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22}'

# 21/12/2013, 14:40
#launch_jobs 20131213_console_output_sklearn_knn localhost:2 'python sklearn_knn.py HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22}'

# 21/12/2013, 14:40
#launch_jobs 20131213_console_output_sklearn_svm localhost:2 'python sklearn_svm.py HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22}'

# 21/12/2013, 15:31
#launch_jobs 20131213_console_output_libsvm localhost:2 'python libsvm.py HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22}'

# 21/12/2013, 14:40
#launch_jobs 20131213_console_output_sklearn_lda_svm localhost:2 'python sklearn_lda_svm.py HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22}'

# 21/12/2013, 14:40
#launch_jobs 20131213_console_output_sklearn_lda_knn localhost:2 'python sklearn_lda_knn.py HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22}'


# 17/12/2013, 10:08
#launch_jobs 20131217_console_output_libsvm_training localhost:2 'python libsvm.py Brats-2_training HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22,24,25,26,27}'
#launch_jobs 20131217_console_output_libsvm_training localhost:2 'python libsvm.py Brats-2_training LG_00{01,02,04,06,08,11,12,13,14,15}'

# 17/12/2013, 14:00
#launch_jobs 20131217_console_output_libsvm_training localhost:2 'python model_libsvm.py BRATS2013/Brats-2_training/  HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22,24,25,26,27} ~/Results/libsvm_results/ interaction.txt allpoints.txt background.txt'
#launch_jobs 20131217_console_output_libsvm_training localhost:2 'python model_libsvm.py BRATS2013/Brats-2_training/  LG_00{01,02,04,06,08,11,12,13,14,15} ~/Results/libsvm_results/ interaction.txt allpoints.txt background.txt'

# 27/01/2014
#launch_jobs 20131217_console_output_libsvm_training localhost:2 'python libsvm.py Brats-2_training HG_0001'

#launch_jobs 20140305_console_output_libsvm_training localhost:2 'python model_libsvm.py BRATS2013/Brats-2_training/ HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22,24,25,26,27} ~/Results/libsvm_results_3dim/ interaction.txt allpoints.txt background.txt'
#launch_jobs 20140305_console_output_libsvm_training localhost:2 'python model_libsvm.py BRATS2013/Brats-2_training/ LG_00{01,02,04,06,08,11,12,13,14,15} ~/Results/libsvm_results_3dim/ interaction.txt allpoints.txt background.txt'


#launch_jobs 20140313_console_output_libsvm_training localhost:2 'python model_libsvm_3d.py BRATS2013/Brats-2_training/ LG_00{01,02,04,06,08,11,12,13,14,15} ~/Results/linear/libsvm_training_3dim/ interaction.txt allpoints.txt background.txt'

#launch_jobs 20140313_console_output_libsvm_training localhost:2 'python model_libsvm_3d.py BRATS2013/Brats-2_challenge/ HG_03{01,02,03,04,05,06,07,08,09,10} ~/Results/linear/libsvm_challenge_3dim/ interaction.txt allpoints.txt background.txt'



#launch_jobs 20140313_console_output_libsvm_training localhost:2 'python model_sklearn_svm.py BRATS2013/Brats-2_training/ HG_00{01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,22} ~/Results/separate_kernel/libsvm_training_6dim/ interaction.txt allpoints.txt background.txt'

launch_jobs 20140313_console_output_libsvm_training localhost:2 'python model_sklearn_svm.py BRATS2013/Brats-2_training/ HG_00{07,24,25,26,27} ~/Results/separate_kernel/libsvm_training_6dim/ interaction.txt allpoints.txt background.txt'

#launch_jobs 20140313_console_output_libsvm_training localhost:2 'python model_sklearn_svm.py BRATS2013/Brats-2_training/ HG_0001 ~/Results/separate_kernel/libsvm_training_6dim/ interaction.txt allpoints.txt background.txt'

#launch_jobs 20140313_console_output_libsvm_training localhost:2 'python model_sklearn_svm.py BRATS2013/Brats-2_training/ LG_00{01,02,04,06,08,11,12,13,14,15} ~/Results/separate_kernel/libsvm_training_6dim/ interaction.txt allpoints.txt background.txt'

#launch_jobs 20140313_console_output_libsvm_training localhost:2 'python model_sklearn_svm.py BRATS2013/Brats-2_challenge/ HG_03{01,02,03,04,05,06,07,08,09,10} ~/Results/separate_kernel/libsvm_challenge_6dim/ interaction.txt allpoints.txt background.txt'




#launch_jobs 20140305_console_output_libsvm_training localhost:2 'python model_libsvm_6d.py BRATS2013/Brats-2_training/ HG_00{10,25} ~/Results/linear/libsvm_training_6dim/ interaction.txt allpoints.txt background.txt'

#launch_jobs 20140313_console_output_libsvm_training localhost:2 'python model_libsvm_6d.py BRATS2013/Brats-2_training/ LG_00{01,04,14} ~/Results/linear/libsvm_training_6dim/ interaction.txt allpoints.txt background.txt'

#launch_jobs 20140313_console_output_libsvm_training localhost:2 'python model_libsvm_6d.py BRATS2013/Brats-2_challenge/ HG_03{01,02,03,04,05,06,07,08,09,10} ~/Results/separate_kernel/libsvm_challenge_6dim/ interaction.txt allpoints.txt background.txt'









