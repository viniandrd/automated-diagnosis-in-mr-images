from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

class Dataset:
    """Helper to return the images (for the chosen modality) and ground truth paths"""
    def __init__(self, path, modality='flair'):
        self.path = path
        self.modality = modality

    def get_data(self):
        "Returns tuples of paths (image, mask)"
        imgs = []
        segs = []
        for path in Path(self.path).rglob('*.png'):
            if 'seg' in str(path):
                segs.append(str(path))
            elif f'{self.modality}' in str(path):
                imgs.append(str(path))

        tuples = list(zip(imgs, segs))
        return tuples


class DataGenerator(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, tuples, img_size=(128, 128), batch_size=4):
        self.input_img_paths = [tuples[i][0] for i in range(len(tuples))]
        self.target_img_paths = [tuples[i][1] for i in range(len(tuples))]

        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, index):
        """Returns tuple (input, target) correspond to batch #idx."""
        x = []
        y = []
        i = index * self.batch_size

        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            img_arr = np.array(img)
            img = np.resize(img_arr, self.img_size)
            img = img.reshape(self.img_size[0], self.img_size[1], 1)
            x.append(img)

        for j, path in enumerate(batch_target_img_paths):
            img = plt.imread(path)
            img = np.array(img)
            img = np.resize(img, self.img_size)
            y.append(tf.one_hot(img.astype(np.int64), 3))

        return np.array(x), np.array(y)
