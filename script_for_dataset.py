import os                   # for path
from imutils import paths   # for collecting image paths
import random               # for random shuffling
import shutil               # to copy images

original_data_path = 'data/original_data'
refined_data_path = 'data/refined_data'     # for training, testing, validataion data

train_path = os.path.sep.join([refined_data_path, 'train'])
test_path = os.path.sep.join([refined_data_path, 'test'])
val_path = os.path.sep.join([refined_data_path, 'val'])

train_split = 0.8   # amount for training
val_split = 0.1     # amount for validation
# testing will be 20% of training

image_paths = list(paths.list_images(original_data_path))   # listing image paths
random.seed(42)
random.shuffle(image_paths)     # shuffle image paths randomly

# split for training and testing
split = int(len(image_paths) * train_split)
new_train_path = image_paths[:split]
new_test_path = image_paths[split:]

# split for validation
split = int(len(new_train_path) * val_split)
new_val_path = new_train_path[:split]
new_train_path = new_train_path[split:]

datasets = [('train', new_train_path, train_path),
            ('val', new_val_path, val_path),
            ('test', new_test_path, test_path)]

for(dType, image_paths, baseOutput) in datasets:
    print('building'.format(dType))
    if not os.path.exists(baseOutput):
        # dir does not exist
        os.makedirs(baseOutput)

    for inputPath in image_paths:
        filename = inputPath.split(os.path.sep)[-1]     # extract filename
        label = filename[-5:-4]                         # extract label
        labelPath = os.path.sep.join([baseOutput, label])   # build path to label dir

        if not os.path.exists(labelPath):
            os.makedirs(labelPath)                      # create labelpath dir

        path = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, path)                      # copy image to path