import tensorflow as tf
from data_manipulation.loaders import *
from pathlib import Path
from config import config as cfg
from tensorflow.keras.preprocessing.image import load_img
import cv2
import numpy as np

class IoU():
    def __init__(self):
        self.path_gts = self.__get_gts()
        self.preds_m1 = [str(path) for path in Path(cfg['predictions']).rglob('*.png') if 'model1' in str(path.parent)]
        self.preds_m2 = [str(path) for path in Path(cfg['predictions']).rglob('*.png') if 'model2' in str(path.parent)]
        self.preds_m3 = [str(path) for path in Path(cfg['predictions']).rglob('*.png') if 'model3' in str(path.parent)]
        self.metric = tf.keras.metrics.MeanIoU(4)

        self.imgs1 = []
        self.imgs2 = []
        self.imgs3 = []
        self.gts = []
        self.__get_imgs()

        self.total_segs = len(self.path_gts)
        self.maiores = self.__get_maiores()

    def __get_maiores(self):
        maiores = []
        m1 = []
        m2 = []
        m3 = []
        for i in range(self.total_segs):
            metricas = self.__get_maior_metrica(i)
            m1.append(metricas[0])
            m2.append(metricas[1])
            m3.append(metricas[2])
            maiores.append(metricas.max())

        dict = {'modelo_1': m1, 'modelo_2': m2, 'modelo_3': m3, 'maior': maiores}
        df = pd.DataFrame(dict)
        df.to_csv('metricas.csv', index=False)

        return maiores

    def __get_maior_metrica(self, index):
        metricas = []

        for i in range(3):
            self.metric.reset_state()
            if i == 0:
                self.metric.update_state(self.gts[index], self.imgs1[index])
                res = self.metric.result().numpy()
                metricas.append(res)

            elif i == 1:
                self.metric.update_state(self.gts[index], self.imgs2[index])
                res = self.metric.result().numpy()
                metricas.append(res)
            else:
                self.metric.update_state(self.gts[index], self.imgs3[index])
                res = self.metric.result().numpy()
                metricas.append(res)

        metricas = np.array(metricas)
        return metricas

    def __get_gts(self):
        df = pd.read_csv('test_set.csv')
        gts_path = df.to_dict()
        gts_path = gts_path['segs']
        gts = [cfg['images_path'] + value for value in gts_path.values()]
        return gts

    def __get_imgs(self):
        print('Reading images...')
        for i in range(len(self.path_gts)):
            img1 = cv2.imread(self.preds_m1[i], 0)
            img1 = cv2.resize(img1, cfg['image_size'])
            self.imgs1.append(img1)

            img2 = cv2.imread(self.preds_m2[i], 0)
            img2 = cv2.resize(img2, cfg['image_size'])
            self.imgs2.append(img2)

            img3 = cv2.imread(self.preds_m3[i], 0)
            img3 = cv2.resize(img3, cfg['image_size'])
            self.imgs3.append(img3)

            gt = cv2.imread(self.path_gts[i], 0)
            gt = cv2.resize(gt, cfg['image_size'])
            self.gts.append(gt)
        print('Done!')

    def iou_result(self, ff, i=0):
        # print(i)
        # img1 = cv2.imread(self.preds_m1[i], 0)
        # img1 = cv2.resize(img1, cfg['image_size'])

        # img2 = cv2.imread(self.preds_m2[i], 0)
        # img2 = cv2.resize(img2, cfg['image_size'])

        # img3 = cv2.imread(self.preds_m3[i], 0)
        # img3 = cv2.resize(img3, cfg['image_size'])

        # gt = cv2.imread(self.path_gts[i], 0)
        # gt = cv2.resize(gt, cfg['image_size'])

        img_res = weighted_image(self.imgs1[i], ff[0]) + weighted_image(self.imgs2[i], ff[1]) + weighted_image(self.imgs3[i], ff[2])

        #img_res = result_image(weighted_image(self.imgs1[i], ff[0]), weighted_image(self.imgs2[i], ff[1]), weighted_image(self.imgs3[i], ff[2]))

        if img_res.max() > 3.0:
            img_res = (img_res / img_res.max()) * 3.0

        img_res_sum = np.sum(img_res)
        array_has_nan = np.isnan(img_res_sum)

        if array_has_nan:
            img_res = np.zeros(img_res.shape, dtype=img_res.dtype)

        img_res = round_image(img_res)
        img_res = img_res.astype(np.uint8)
        self.metric.reset_state()
        self.metric.update_state(self.gts[i], img_res)
        res = self.metric.result().numpy()
        return res