import tensorflow as tf
from data_manipulation.loaders import *
from pathlib import Path
from config import config as cfg
from tensorflow.keras.preprocessing.image import load_img

class CustomMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


class IoU():
    def __init__(self):
        self.preds_m2 = [path for path in Path(cfg['predictions']).rglob('*.png') if 'model2' in str(path.parent)]
        self.preds_m3 = [path for path in Path(cfg['predictions']).rglob('*.png') if 'model3' in str(path.parent)]
        self.gts = self._get_gts()

        self.total_segs = len(self.gts)

    def _get_gts(self):
        df = pd.read_csv('test_set.csv')
        gts_path = df.to_dict()
        gts_path = gts_path['segs']
        gts = [cfg['images_path'] + value for value in gts_path.values()]
        return gts

    def iou_result(self, ff):
        metric = tf.keras.metrics.MeanIoU(cfg['classes'])
        metrics = []
        for i in range(self.total_segs):
            pred_m2 = load_img(self.preds_m2[i], target_size=cfg['image_size'], color_mode="grayscale")
            pred_m2 = np.asarray(pred_m2)

            pred_m3 = load_img(self.preds_m3[i], target_size=cfg['image_size'], color_mode="grayscale")
            pred_m3 = np.asarray(pred_m3)

            gt = load_img(self.gts[i], target_size=(128,128), color_mode="grayscale")
            gt = np.asarray(gt)

            img_res = result_image(weighted_image(pred_m2, ff[0]), weighted_image(pred_m3, ff[1]))
            metrics.append(metric.update_state(gt, img_res))

        metrics = np.array(metrics)
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
