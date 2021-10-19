from pathlib import Path

import cv2, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import save_img
import imageio
import pandas as pd
from utils import *
import nibabel

class Dataset:
    def __init__(self, path, images_path, modality='flair', initial_slice=0, final_slice=155, train_split=0.8, test_split=0.1, val_split=0.1, ):
        self.path = path
        self.images_path = images_path
        self.initial_slice = initial_slice
        self.final_slice = final_slice
        self.modality = modality

        self.train_split = train_split
        self.test_split = train_split + test_split
        self.val_split = train_split + val_split

        self.train = None
        self.val = None
        self.test = None

        self.patients_HGG = []
        self.patients_LGG = []

        self._split_data()
        self._save_sets_csv()

    def _extract(self):
        amount = count(self.input_path, 'nii.gz')
        c = 0
        for path in Path(self.input_path).rglob('*.nii.gz'):
            HGG = False

            data = nibabel.load(path)
            image_array = data.get_fdata()

            # Handling patient folders
            if path.parts[-3] == 'HGG':
                HGG = True
                if path.parts[-2] not in self.patients_HGG:
                    self.patients_HGG.append(path.parts[-2])

            else:
                if path.parts[-2] not in self.patients_LGG:
                    self.patients_LGG.append(path.parts[-2])

            if HGG:
                output_label = self.output_path + 'HGG/'
                index = self.patients_HGG.index(path.parts[-2])
            else:
                output_label = self.output_path + 'LGG/'
                index = self.patients_LGG.index(path.parts[-2])

            patient = 'patient{:03d}'.format(index + 1)

            for slice in range(self.initial_slice, self.final_slice):
                img = image_array[:, :, slice]
                img = np.rot90(np.rot90(np.rot90(img)))
                img = resize(img, (240, 240, 3), order=0, preserve_range=True, anti_aliasing=False)

                if ('seg' in path.parts[-1]):
                    img[img == 1] = 1
                    img[img == 2] = 2
                    img[img == 4] = 3
                    output_tissue = output_label + 'seg/'

                    if not os.path.exists(output_tissue):
                        os.makedirs(output_tissue)
                        print("Created ouput directory: " + output_tissue)

                    mask_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    # not black image
                    if not cv2.countNonZero(mask_gray) == 0:
                        imageio.imwrite(output_tissue + patient + '_slice{:03d}'.format(slice) + '.png', img.astype(np.uint8))


                if ('flair' in path.parts[-1]):
                    output_tissue = output_label + 'flair/'

                    if not os.path.exists(output_tissue):
                        os.makedirs(output_tissue)
                        print("Created ouput directory: " + output_tissue)

                    save_img(output_tissue + patient + '_slice{:03d}'.format(slice) + '.png', img, scale=True)

                if ('t1' in path.parts[-1]):
                    output_tissue = output_label + 't1/'

                    if not os.path.exists(output_tissue):
                        os.makedirs(output_tissue)
                        print("Created ouput directory: " + output_tissue)

                    save_img(output_tissue + patient + '_slice{:03d}'.format(slice) + '.png', img, scale=True)

                if ('t1ce' in path.parts[-1]):
                    output_tissue = output_label + 't1c/'

                    if not os.path.exists(output_tissue):
                        os.makedirs(output_tissue)
                        print("Created ouput directory: " + output_tissue)

                    save_img(output_tissue + patient + '_slice{:03d}'.format(slice) + '.png', img, scale=True)

                if ('t2' in path.parts[-1]):
                    output_tissue = output_label + 't2/'
                    if not os.path.exists(output_tissue):
                        os.makedirs(output_tissue)
                        print("Created ouput directory: " + output_tissue)

                    imageio.imwrite(output_tissue + patient + '_slice{:03d}'.format(slice) + '.png', img)

            printProgressBar(c + 1, amount, prefix='Progress:', suffix='Complete', length=50)
            c += 1

    def _split_data(self):
        print('>> Splitting data....')
        imgs = []
        segs = []
        for path in Path(self.path).rglob('*.png'):
            if 'seg' in str(path):
                segs.append(str(path))
            elif f'{self.modality}' in str(path):
                imgs.append(str(path))

        tuples = list(zip(imgs, segs))

        self.train = tuples[:int(len(tuples) * self.train_split)]
        self.val = tuples[int(len(tuples) * self.train_split):int(len(tuples) * self.test_split)] 
        self.test = tuples[int(len(tuples) * self.val_split):]
        print('!! Split done!')


    def _save_sets_csv(self):
        imgs = []
        segs = []

        for i in range(len(self.test)):
            imgs.append(self.test[i][0])
            segs.append(self.test[i][1])
    
        data_test = {'imgs': imgs, 'segs': segs}
        df_test = pd.DataFrame(data_test)
        df_test.to_csv('test_set.csv', index=False)
        print('!! Test set paths saved to > test_set.csv')

    def get_data(self):
        return self.train, self.val, self.test


    


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, tuples, img_size=(128, 128), batch_size=1):
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
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
            y.append(tf.one_hot(img.astype(np.int64), 4))

        return np.array(x), np.array(y)
