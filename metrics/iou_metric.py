from data_manipulation.loaders import *
from pathlib import Path
from config import config as cfg


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
        acc_iou = 0
        for index in range(self.total_segs):
            pred_m2 = read_img(str(self.preds_m2[index]), cfg['image_size'])
            pred_m3 = read_img(str(self.preds_m3[index]), cfg['image_size'])
            gt = read_img(str(self.gts[index]), cfg['image_size'])

            img_res = result_image(weighted_image(pred_m2, ff[0]), weighted_image(pred_m3, ff[1]))
            acc_iou = acc_iou + self.iou_coef(img_res, gt)

        avg_iou = acc_iou / self.total_segs
        return avg_iou

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
