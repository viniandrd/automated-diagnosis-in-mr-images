config = dict()

# BraTS19 dataset path (extracted once)
config['dataset_path'] = ''

# Path for images extraction (if not exists, will be created)
config['images_path'] = r'D:\Dados\ViniAndrade\TCC\BraTS19\BraTS19 PNG/'

# Which MR modality extract
config['modality'] = 'flair'

# Extract slices starting from
config['initial_slice'] = 70

# Extract slices until
config['final_slice'] = 120

# Input shape for segmentation_models
config['input_shape'] = (None, None, 1)

# Epochs for training models
config['epochs'] = 100

# Batch size
config['batch_size'] = 8

# Path to save results (weights, training history, graphics)
config['save_dir'] = ''

# Input shape of the models
config['image_size'] = (128, 128)

# Classes of the mask
config['classes'] = 4

# Path to folder where it contains predictions from model2
config['predictions'] = './predictions/'
