import cv2
import torch
import numpy as np

# Additional Scripts
from train_transunet import TransUNetSeg

from utils.utils import thresh_func
from config import cfg


class SegInference:
    def __init__(self, model_path, device):
        self.device = device
        self.transunet = TransUNetSeg(device)
        self.transunet.load_model(model_path)

    def read_and_preprocess(self, p):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        img_torch = cv2.resize(img, (cfg.transunet.img_dim, cfg.transunet.img_dim))
        img_torch = img_torch / 255.
        img_torch = img_torch.transpose((2, 0, 1))
        img_torch = np.expand_dims(img_torch, axis=0)
        img_torch = torch.from_numpy(img_torch.astype('float32')).to(self.device)

        return img, img_torch, h, w

    def infer(self, path, merged=False):
        path = [path] if isinstance(path, str) else path

        preds = []
        for p in path:
            print(p)
            img, img_torch, orig_h, orig_w = self.read_and_preprocess(p)
            with torch.no_grad():
                pred_mask = self.transunet.model(img_torch)
                pred_mask = torch.sigmoid(pred_mask)
                pred_mask = pred_mask.detach().cpu().numpy().transpose((0, 2, 3, 1))

                pred_mask = cv2.resize(pred_mask[0, ...], (orig_w, orig_h))
                pred_mask = thresh_func(pred_mask, thresh=0.75)

                if merged:
                    pred_mask = cv2.bitwise_and(img, img, mask=pred_mask.astype('uint8'))

                preds.append(pred_mask)

        return preds


if __name__ == '__main__':
    inf = SegInference('transunet.pt', 'cuda:0')

    import os

    paths = os.listdir('test/img')
    paths = [f'img/{p}' for p in paths]
    inf.infer(paths)
