from config import config as cfg
from custom_objects.custom_layer import *
from custom_objects.custom_loss import *
from custom_objects.custom_metrics import *
from models.model1 import *
from models.model2 import *
from models.model3 import *
import datetime
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


data = Dataset(cfg['dataset_path'], cfg['images_path'], 
                cfg['modality'], initial_slice=cfg['initial_slice'], 
                final_slice=cfg['final_slice'], extract=cfg['extract'])

train_gen = DataGenerator(data.train, batch_size=cfg['batch_size'])
val_gen = DataGenerator(data.val, batch_size=cfg['batch_size'])

steps_per_epoch_train = (len(data.train)) // cfg['batch_size']
steps_per_epoch_val = (len(data.val)) // cfg['batch_size']


tf.keras.backend.clear_session()


log_dir = cfg['logs_dir'] + 'model' + str(cfg['model']) + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

if cfg['model'] == 1:
    model = Unet_model1(img_shape=cfg['image_size'] + (1,))
elif cfg['model'] == 2:
    model = Unet_model2(img_shape=cfg['image_size'] + (1,))
else:
    model = unet_backbone_resnet34_bce_jaccard_loss()

history = model.fit(train_gen, validation_data=val_gen, 
                    steps_per_epoch=steps_per_epoch_train, validation_steps = steps_per_epoch_val,
                    epochs = cfg['epochs'], callbacks=[tensorboard_callback])

weights_dir = cfg['save_dir'] + 'weights/'
model.save(weights_dir + 'model' + str(cfg['model']) + '.h5')