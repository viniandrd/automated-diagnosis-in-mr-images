from pathlib import Path

import cv2
import glob
import imageio
import nibabel
import numpy as np
import os
import pandas as pd
import random
import sys
import tensorflow as tf
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img

from config import config as cfg


class Dataset:
    def __init__(self, path, images_path, modality='flair', extract=True, initial_slice=0, final_slice=155,
                 train_split=0.8,
                 test_split=0.1, val_split=0.1):
        self.input_path = path
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

        if extract:
            self.patients_HGG = []
            self.patients_LGG = []
            self._extract()

        self._filter()
        self._get_sets()

    def _get_csvs(self):
        print('>> Searching for sets in csv format..')
        sets = dict()
        try:
            train_set = pd.read_csv('train_set.csv')
            val_set = pd.read_csv('val_set.csv')
            test_set = pd.read_csv('test_set.csv')
            sets['train_set'] = train_set
            sets['val_set'] = val_set
            sets['test_set'] = test_set
            print('!! Found them.')
            return sets
        except FileNotFoundError:
            print("!! Sets not found!")
            return sets

    def join_imgs_path(self, tuples):
        for idx, tup in enumerate(tuples):
            # print(f'{idx}: {tup}')
            img = self.images_path + tup[0]
            seg = self.images_path + tup[1]
            tuples[idx][0] = img
            tuples[idx][1] = seg
        return tuples

    def _get_sets(self):
        sets = self._get_csvs()
        train = None
        val = None
        test = None

        if len(sets) == 0:
            self._split_data()
            self._save_sets_csv()
        else:
            for set in sets:
                if 'train' in set:
                    tuples = sets['train_set'].values.tolist()
                    tuples = self.join_imgs_path(tuples)
                    self.train = tuples
                elif 'val' in set:
                    tuples = sets['val_set'].values.tolist()
                    tuples = self.join_imgs_path(tuples)
                    self.val = tuples
                else:
                    tuples = sets['test_set'].values.tolist()
                    tuples = self.join_imgs_path(tuples)
                    self.test = tuples

    def _split_data(self):
        print('>> Splitting data....')
        imgs = []
        segs = []
        for path in Path(self.images_path).rglob('*.png'):
            path = str(path)
            if 'seg' in path:
                # segs.append(str(path[-31:]))
                segs.append(str(path))
            elif f'{self.modality}' in path:
                # imgs.append(str(path[-33:]))
                imgs.append(str(path))

        tuples = list(zip(imgs, segs))
        random.shuffle(tuples)
        self.train = tuples[:int(len(tuples) * self.train_split)]
        self.val = tuples[int(len(tuples) * self.train_split):int(len(tuples) * self.test_split)]
        self.test = tuples[int(len(tuples) * self.val_split):]
        print('<< Split done!')

    def _save_sets_csv(self):
        sets = [self.train, self.val, self.test]

        idx = 0
        for set in sets:
            imgs = []
            segs = []

            for i in range(len(set)):
                img = set[i][0]
                seg = set[i][1]
                imgs.append(img[-33:])
                segs.append(seg[-31:])

            data = {'imgs': imgs, 'segs': segs}
            df_test = pd.DataFrame(data)
            if idx == 0:
                df_test.to_csv('train_set.csv', index=False)
                print('!! Train set paths saved to > train_set.csv')
            elif idx == 1:
                df_test.to_csv('val_set.csv', index=False)
                print('!! Val set paths saved to > val_set.csv')
            else:
                df_test.to_csv('test_set.csv', index=False)
                print('!! Test set paths saved to > test_set.csv')
            idx += 1

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
                output_label = self.images_path + 'HGG/'
                index = self.patients_HGG.index(path.parts[-2])
            else:
                output_label = self.images_path + 'LGG/'
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

                    mask_gray = cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2GRAY)
                    # not black image
                    if not cv2.countNonZero(mask_gray) == 0:
                        imageio.imwrite(output_tissue + patient + '_slice{:03d}'.format(slice) + '.png',
                                        img.astype(np.uint8))

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

                    save_img(output_tissue + patient + '_slice{:03d}'.format(slice) + '.png', img, scale=True)

            printProgressBar(c + 1, amount, prefix='Progress:', suffix='Complete', length=50)
            c += 1

    def _filter(self):
        print('>> Filtering black images (without ground truth)..')
        labels = ['HGG', 'LGG']
        for label in labels:
            path_to_tumors = self.images_path + f'{label}/seg'
            path_to_modality = self.images_path + f'{label}/flair'

            tumors = glob.glob(path_to_tumors + '/*.png')
            images = glob.glob(path_to_modality + '/*.png')

            tumors_filter = []
            images_filters = []

            idx = 0
            for tumor in tumors:
                tumors_filter.append(tumor[-16:])
                idx += 1

            idx = 0
            for img in images:
                images_filters.append(img[-16:])
                idx += 1

            count = 0
            idx = 0
            for imgs in images_filters:
                if not imgs in tumors_filter:
                    os.remove(images[idx])
                    count += 1
                idx += 1

            count2 = 0
            for img in images_filters:
                if not img in tumors_filter:
                    count2 += 1

        print('<< Done!\n')

    def get_data(self):
        return self.train, self.val, self.test


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, tuples, img_size=(128, 128), batch_size=8, classes=4):
        # random.shuffle(tuples)
        self.input_img_paths = [tuples[i][0] for i in range(len(tuples))]
        self.target_img_paths = [tuples[i][1] for i in range(len(tuples))]
        self.classes = classes
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
            img_arr = img_arr / float(255)
            img = np.resize(img_arr, self.img_size)
            img = img.reshape(self.img_size[0], self.img_size[1], 1)
            maior = np.max(img) if np.max(img) > 0 else 1
            img = img / maior
            x.append(img)

        for j, path in enumerate(batch_target_img_paths):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
            y.append(tf.one_hot(img.astype(np.int64), self.classes))

        return np.array(x), np.array(y)


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def count(input, type):
    count = 0
    for path in Path(input).rglob('*.' + type):
        count += 1

    return count


np.set_printoptions(threshold=sys.maxsize)


def read_img(p, img_size):
    """
    :param p: path of image
    :return: image as numpy array
    """
    img = cv2.imread(p)
    img = cv2.resize(img, img_size)
    img_c = np.copy(np.asarray(img)).astype(np.float)
    return img_c


def weighted_image(img, w):
    """
    :param img: image (numpy array)
    :param w: weight of firefly
    :return: new image
    """
    img_c = np.copy(np.asarray(img)).astype(np.float)

    lin = np.size(img_c, 0)
    col = np.size(img_c, 1)
    for n in range(lin):
        for m in range(col):
            img_c[n][m] = img_c[n][m] * w
            #for k in range(len(img_c[n][m])):
                #img_c[n][m][k] = img_c[n][m][k] * w

    return img_c


def result_image(img1, img2):
    lin = np.size(img1, 0)
    col = np.size(img1, 1)
    res = np.zeros(cfg['image_size'] + (3,))

    for n in range(lin):
        for m in range(col):
            res[n][m] = img1[n][m] + img2[n][m]
            #for k in range(len(res[n][m])):
                #res[n][m][k] = img1[n][m][k] + img2[n][m][k]
    return res
