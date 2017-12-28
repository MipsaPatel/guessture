import os

from sklearn.model_selection import train_test_split

from .helper import get_target, is_video
from .params import Parameters

# load parameters
P = Parameters()

# change the working directory for easier file access
os.chdir(P.data_dir)

print('Loading files...')
# load only video files
files = sorted(filter(is_video, os.listdir('.')))
# compute targets
target = list(map(get_target, files))

# create folders for train and test data
print('Creating folders...')
os.makedirs(P.train_data)
os.makedirs(P.test_data)

# split based on target
print('Splitting data...')
train_files, test_files = train_test_split(files, test_size=P.test_size, random_state=1048, stratify=target)

# move files to respective folders
print('Moving...')
os.system('mv -t %s ' % P.train_data + ' '.join(train_files))
os.system('mv -t %s ' % P.test_data + ' '.join(test_files))
