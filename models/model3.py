from typing import Callable
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers.schedules import *
import segmentation_models as sm
from custom_objects.custom_metrics import CustomMeanIOU
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from config import config as cfg


SM_FRAMEWORK=tf.keras

def unet_backbone_resnet34_bce_jaccard_loss():
    unet_model = sm.Unet('resnet34', classes=cfg['classes'], activation='softmax',input_shape=cfg['input_shape'], encoder_weights=None)

    unet_model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[CustomMeanIOU(cfg['classes'], dtype=np.float32),sm.metrics.f1_score],
    )
    return unet_model