config = dict()

# BraTS19 dataset path (extracted once)
config['dataset_path'] = 'D:/ViniAndrade/TCC/ConjuntodeDados/BraTS19/training/'

# Path for images extraction
config['images_path'] = 'D:/ViniAndrade/TCC/ConjuntodeDados/BraTS19 PNG 70-120/'

# Which MR modality extract
config['modality'] = 'flair'

# Input shape for segmentation_models
config['input_shape'] = (None, None, 1)

# Epochs for training models
config['epochs'] = 200

# Batch size
config['batch_size'] = 32

# Path to save results (weights, training history, graphics)
config['save_dir'] = ''

# Input shape of the models
config['image_shape'] = (128, 128)

# Classes of the mask
config['classes'] = 4

# Ouput for saving ground truth of test set
config['out_test'] = './test_set/'