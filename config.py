config = dict()

# BraTS19 dataset path (extracted once)
config['dataset_path'] = 'D:/ViniAndrade/TCC/ConjuntodeDados/BraTS19/training/'

# Need to extract .nii files?
config['extract'] = True

# Path for PNG images (if not exists, will be created)
config['images_path'] = ''

# Which MR modality extract or which is extracted
config['modality'] = 'flair'

# Extract slices starting from
config['initial_slice'] = 70

# Extract slices until
config['final_slice'] = 120

# Input shape for segmentation_models
config['input_shape'] = (None, None, 1)

# Epochs for training models
config['epochs'] = 300

# Batch size
config['batch_size'] = 8

# Path to save results (weights, training history, graphics)
config['save_dir'] = './results/'

# Path to save logs file (for Tensorboard only)
config['logs_dir'] = './results/logs/'

# Which model to train (1, 2 or 3)
config['model'] = 1 

# Input shape of the models
config['image_size'] = (128, 128)

# Classes of the mask
config['classes'] = 4

# Path to folder where it contains predictions from model2
config['predictions'] = 'E:/predictions/'
