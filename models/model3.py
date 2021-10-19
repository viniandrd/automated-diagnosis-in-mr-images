from typing import Callable
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers.schedules import *
import segmentation_models as sm
from metrics.custom_metrics import CustomMeanIOU
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from config import config as cfg


SM_FRAMEWORK=tf.keras


def unet_backbone_resnet34_jaccard_loss():
    unet_model = sm.Unet('resnet34', classes=cfg['classes'], activation='softmax',input_shape=cfg['input_shape'], encoder_weights=None)

    for layer in unet_model.layers:
        if hasattr(layer,'activation'):
            layer.activation = LeakyReLU(alpha=0.01)

    unet_model.compile(
        'Adam',
        loss=sm.losses.jaccard_loss,
        metrics=[CustomMeanIOU(cfg['classes'], dtype=np.float32),sm.metrics.f1_score],
    )
    return unet_model

def unet_backbone_resnet34_bce_jaccard_loss_relu():
    unet_model = sm.Unet('resnet34', classes=cfg['classes'], activation='softmax',input_shape=cfg['input_shape'], encoder_weights=None)

    unet_model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[CustomMeanIOU(cfg['classes'], dtype=np.float32),sm.metrics.f1_score],
    )
    return unet_model

def unet_backbone_resnet34_jaccard_loss_relu():
    unet_model = sm.Unet('resnet34', classes=cfg['classes'], activation='softmax',input_shape=cfg['input_shape'], encoder_weights=None)

    unet_model.compile(
        'Adam',
        loss=sm.losses.jaccard_loss,
        metrics=[CustomMeanIOU(cfg['classes'], dtype=np.float32),sm.metrics.f1_score],
    )
    return unet_model

def unet_jaccard_loss():
    unet_model = sm.Unet(classes=cfg['classes'], activation='softmax',input_shape=cfg['input_shape'], encoder_weights=None)

    for layer in unet_model.layers:
     if hasattr(layer,'activation'):
         layer.activation = LeakyReLU(alpha=0.01)

    unet_model.compile(
        'Adam',
        loss=sm.losses.jaccard_loss,
        metrics=[CustomMeanIOU(cfg['classes'], dtype=np.float32),sm.metrics.f1_score],
    )
    return unet_model

def unet_jaccard_loss_relu():
    unet_model = sm.Unet(classes=cfg['classes'], activation='softmax',input_shape=cfg['input_shape'], encoder_weights=None)

    unet_model.compile(
        'Adam',
        loss=sm.losses.jaccard_loss,
        metrics=[CustomMeanIOU(cfg['classes'], dtype=np.float32),sm.metrics.f1_score],
    )
    return unet_model

def unet_dice_loss():
    unet_model = sm.Unet(classes=cfg['classes'], activation='softmax',input_shape=cfg['input_shape'], encoder_weights=None)

    for layer in unet_model.layers:
     if hasattr(layer,'activation'):
         layer.activation = LeakyReLU(alpha=0.01)

    unet_model.compile(
        'Adam',
        loss=sm.losses.dice_loss,
        metrics=[CustomMeanIOU(cfg['classes'], dtype=np.float32),sm.metrics.f1_score],
    )

    return unet_model

def unet_dice_loss_relu():
    unet_model = sm.Unet(classes=cfg['classes'], activation='softmax',input_shape=cfg['input_shape'], encoder_weights=None)

    unet_model.compile(
        'Adam',
        loss=sm.losses.dice_loss,
        metrics=[CustomMeanIOU(cfg['classes'], dtype=np.float32),sm.metrics.f1_score],
    )

    return unet_model

def unet_jaccard_loss():
    unet_model = sm.Unet(classes=cfg['classes'], activation='softmax',input_shape=cfg['input_shape'], encoder_weights=None)

    for layer in unet_model.layers:
     if hasattr(layer,'activation'):
         layer.activation = LeakyReLU(alpha=0.01)

    unet_model.compile(
        'Adam',
        loss=sm.losses.jaccard_loss,
        metrics=[CustomMeanIOU(cfg['classes'], dtype=np.float32),sm.metrics.f1_score],
    )

    return unet_model

def unet_bce_jaccard_loss():
    unet_model = sm.Unet(classes=cfg['classes'], activation='softmax',input_shape=cfg['input_shape'], encoder_weights=None)

    for layer in unet_model.layers:
     if hasattr(layer,'activation'):
         layer.activation = LeakyReLU(alpha=0.01)

    unet_model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[CustomMeanIOU(cfg['classes'], dtype=np.float32),sm.metrics.f1_score],
    )

    return unet_model

def unet_bce_jaccard_loss_relu():
    unet_model = sm.Unet(classes=cfg['classes'], activation='softmax',input_shape=cfg['input_shape'], encoder_weights=None)

    unet_model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[CustomMeanIOU(cfg['classes'], dtype=np.float32),sm.metrics.f1_score],
    )

    return unet_model

def unet_backbone_resnet34_dice_loss():
    unet_model = sm.Unet('resnet34', classes=cfg['classes'], activation='softmax',input_shape=cfg['input_shape'], encoder_weights=None)

    for layer in unet_model.layers:
        if hasattr(layer,'activation'):
            layer.activation = LeakyReLU(alpha=0.01)

    unet_model.compile(
        'Adam',
        loss=sm.losses.dice_loss,
        metrics=[sm.metrics.f1_score,CustomMeanIOU(cfg['classes'], dtype=np.float32)],
    )

    return unet_model

def unet_backbone_resnet34_dice_loss_relu():
    unet_model = sm.Unet('resnet34', classes=cfg['classes'], activation='softmax',input_shape=cfg['input_shape'], encoder_weights=None)

    unet_model.compile(
        'Adam',
        loss=sm.losses.dice_loss,
        metrics=[sm.metrics.f1_score,CustomMeanIOU(cfg['classes'], dtype=np.float32)],
    )

    return unet_model

MODELS: dict[str, Callable[[], Model]]  = {
    'unet_backbone_resnet34_jaccard_loss':unet_backbone_resnet34_jaccard_loss,
    'unet_jaccard_loss':unet_jaccard_loss,
    'unet_dice_loss':unet_dice_loss,
    'unet_backbone_resnet34_dice_loss':unet_backbone_resnet34_dice_loss,
    'unet_backbone_resnet34_dice_loss_relu':unet_backbone_resnet34_dice_loss_relu,
    'unet_bce_jaccard_loss':unet_bce_jaccard_loss,
    'unet_backbone_resnet34_jaccard_loss_relu':unet_backbone_resnet34_jaccard_loss_relu,
    'unet_backbone_resnet34_bce_jaccard_loss_relu':unet_backbone_resnet34_bce_jaccard_loss_relu,
    'unet_dice_loss_relu':unet_dice_loss_relu,
    'unet_jaccard_loss_relu':unet_jaccard_loss_relu,
    'unet_bce_jaccard_loss_relu':unet_bce_jaccard_loss_relu
}