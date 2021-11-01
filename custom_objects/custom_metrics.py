import tensorflow as tf
from data_manipulation.loaders import *
from pathlib import Path
from config import config as cfg
from tensorflow.keras.preprocessing.image import load_img
import cv2

class CustomMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


class IoU():
    def __init__(self):
        self.preds_m1 = [str(path) for path in Path(cfg['predictions']).rglob('*.png') if 'model1' in str(path.parent)]
        self.preds_m2 = [str(path) for path in Path(cfg['predictions']).rglob('*.png') if 'model2' in str(path.parent)]
        self.preds_m3 = [str(path) for path in Path(cfg['predictions']).rglob('*.png') if 'model3' in str(path.parent)]
        self.gts = self._get_gts()

        self.total_segs = len(self.gts)

    def _get_gts(self):
        df = pd.read_csv('test_set.csv')
        gts_path = df.to_dict()
        gts_path = gts_path['segs']
        gts = [cfg['images_path'] + value for value in gts_path.values()]
        return gts

    def iou_result(self, ff):
        metrics = []
        metric = CustomMeanIOU(cfg['classes'])
        for i in range(self.total_segs):
            if i % 500 == 0:
                print(i)

            pred_m1 = cv2.imread(self.preds_m1[i], cv2.IMREAD_GRAYSCALE)
            pred_m1 = cv2.resize(pred_m1, cfg['image_size'])

            pred_m2 = cv2.imread(self.preds_m2[i], cv2.IMREAD_GRAYSCALE)
            pred_m2 = cv2.resize(pred_m2, cfg['image_size'])

            pred_m3 = cv2.imread(self.preds_m3[i], cv2.IMREAD_GRAYSCALE)
            pred_m3 = cv2.resize(pred_m3, cfg['image_size'])

            gt = cv2.imread(self.gts[i], cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, cfg['image_size'])
            gt = tf.one_hot(gt.astype(np.int64), 4)

            img_res = result_image(weighted_image(pred_m1, ff[0]), weighted_image(pred_m2, ff[1]), weighted_image(pred_m3, ff[2]))
            img_res = tf.one_hot(img_res.astype(np.int64), 4)
            metric.update_state(gt, img_res)
            metrics.append(metric.result().numpy())

        metrics = np.array(metrics)
        print('--')
        print(f'Weights: {ff}')
        print(f'Avg meanIoU: {metrics.mean()}')

        return metrics.mean()

    def iou_coef(self, img_pred, img_true):
        lin = np.size(img_pred, 0)
        col = np.size(img_pred, 1)

        total_pixel = lin * col
        acc = 0

        for n in range(lin):
            for m in range(col):
                if np.array_equal(img_pred[n][m], img_true[n][m]):
                    acc += 1

        iou = acc / total_pixel
        return iou
