import zipfile,os.path
def unzip(source_filename,name2):
    dirname = os.path.dirname(source_filename)
    zfile = zipfile.ZipFile(source_filename)
    for name in zfile.namelist():
      out_put_path = dirname+'/'+name
      fd = open(out_put_path,"w")
      fd.write(zfile.read(name))
      fd.close()
    os.rename(out_put_path,name2)




import os

 #for roots,dirs,files in os.walk(os.getcwd()):
for brain in os.listdir(os.getcwd()):
    brain_root = os.getcwd()+'/'+brain
    #all_files = [f for f in os.path.listdir(brain_root)]
    zip_allpoints = brain_root + '/' + 'allpoints_'+brain+'.zip'                          
    dis_allpoints = brain_root + '/' + 'allpoints.txt'
    unzip(zip_allpoints, dis_allpoints)

    zip_background = brain_root + '/' + 'background_'+brain+'.zip'                          
    dis_background = brain_root + '/' + 'background.txt'
    unzip(zip_background, dis_background)

    zip_train = brain_root + '/' + 'interaction_'+brain+'.zip'                          
    dis_train = brain_root + '/' + 'interaction.txt'
    unzip(zip_train, dis_train)

for brain in os.listdir(os.getcwd()):
    brain_root = os.getcwd()+'/'+brain
    #all_files = [f for f in os.path.listdir(brain_root)]
    zip_allpoints = brain_root + '/' + 'allpoints_'+brain+'.zip'                          
    dis_allpoints = brain_root + '/' + 'allpoints.txt'
    #unzip(zip_allpoints, dis_allpoints)

    zip_background = brain_root + '/' + 'background_'+brain+'.zip'                          
    dis_background = brain_root + '/' + 'background.txt'
    #unzip(zip_background, dis_background)

    zip_train = brain_root + '/' + 'interaction_'+brain+'.zip'                          
    dis_train = brain_root + '/' + 'interaction.txt'
    #unzip(zip_train, dis_train)
    os.remove(zip_allpoints)
    os.remove(zip_background)
    os.remove(zip_train)




