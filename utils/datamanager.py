import os
import kagglehub
import shutil

# import download
from random import shuffle

def download_dataset(dest_path):

    # Download dataset using kagglehub (this caches it)
    path = kagglehub.dataset_download("yassershrief/hockey-fight-vidoes")

    # Copy all contents to current working directory
    if os.path.isdir(path):
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(dest_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    print("Dataset copied to:", dest_path)


#function to label video names depending of its names
def label_video_names(in_dir):
    
    # list containing video names
    names = []
    # list containin video labels [1, 0] if it has violence and [0, 1] if not
    labels = []
    
    
    for current_dir, dir_names,file_names in os.walk(in_dir):
        
        for file_name in file_names:
            
            if file_name[0:2] == 'fi':
                labels.append([1,0])
                names.append(file_name)
            elif file_name[0:2] == 'no':
                labels.append([0,1])
                names.append(file_name)
                     
            
    c = list(zip(names,labels))
    # Suffle the data (names and labels)
    shuffle(c)
    
    names, labels = zip(*c)
            
    return names, labels