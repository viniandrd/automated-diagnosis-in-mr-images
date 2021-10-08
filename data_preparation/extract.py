import os
from pathlib import Path

import imageio
import nibabel
import numpy as np
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import save_img

from utils import printProgressBar, count


def extract(input, output, initial_slice=0, final_slice=155):
    patients_HGG = []
    patients_LGG = []

    amount = count(input, 'nii.gz')
    c = 0
    for path in Path(input).rglob('*.nii.gz'):
        HGG = False
        data = nibabel.load(path)
        image_array = data.get_fdata()

        # Handling patient folders
        if path.parts[-3] == 'HGG':
            HGG = True
            if path.parts[-2] not in patients_HGG:
                patients_HGG.append(path.parts[-2])

        else:
            if path.parts[-2] not in patients_LGG:
                patients_LGG.append(path.parts[-2])

        if HGG:
            output_label = output + 'HGG/'
            index = patients_HGG.index(path.parts[-2])
        else:
            output_label = output + 'LGG/'
            index = patients_LGG.index(path.parts[-2])

        patient = 'patient{:03d}'.format(index + 1)

        for slice in range(initial_slice, final_slice):
            img = image_array[:, :, slice]
            img = np.rot90(np.rot90(np.rot90(img)))
            img = resize(img, (240, 240), order=0, preserve_range=True, anti_aliasing=False)

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

            if ('seg' in path.parts[-1]):
                img[img == 1] = 1
                img[img == 2] = 2
                img[img == 4] = 3
                output_tissue = output_label + 'seg/'

                if not os.path.exists(output_tissue):
                    os.makedirs(output_tissue)
                    print("Created ouput directory: " + output_tissue)

                imageio.imwrite(output_tissue + patient + '_slice{:03d}'.format(slice) + '.png', img.astype(np.uint8))

        printProgressBar(c + 1, amount, prefix='Progress:', suffix='Complete', length=50)
        c += 1

    print('Done!')
