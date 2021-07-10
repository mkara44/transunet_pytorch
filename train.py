from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Additional Scripts
from utils import transforms as T
from utils.dataset import DentalDataset
from utils.utils import EpochCallback

from config import cfg

from train_transunet import TransUNetSeg


class TrainTestPipe:
    def __init__(self, train_path, test_path, device):
        self.device = device

        self.train_loader = self.__load_dataset(train_path, train=True)
        self.test_loader = self.__load_dataset(test_path, train=True)

        self.transunet = TransUNetSeg(self.device)

    def __load_dataset(self, path, train=False):
        shuffle = False
        transform = False

        if train:
            shuffle = True
            transform = transforms.Compose([T.BGR2RGB(),
                                            T.Rescale(cfg.transunet.img_dim),
                                            T.RandomAugmentation(2),
                                            T.Normalize(),
                                            T.ToTensor()])
        set = DentalDataset(path, transform)
        loader = DataLoader(set, batch_size=cfg.batch_size, shuffle=shuffle)

        return loader

    def __loop(self, loader, step_func, t):
        total_loss = 0

        for step, data in enumerate(loader):
            img, mask = data['img'], data['mask']
            img = img.to(self.device)
            mask = mask.to(self.device)

            loss, cls_pred = step_func(img=img, mask=mask)

            total_loss += loss

            t.update()

        return total_loss

    def train(self):
        callback = EpochCallback(cfg.model_name, cfg.epoch,
                                 self.transunet.model, self.transunet.optimizer, 'test_loss', cfg.patience)

        for epoch in range(cfg.epoch):
            with tqdm(total=len(self.train_loader) + len(self.test_loader)) as t:
                train_loss = self.__loop(self.train_loader, self.transunet.train_step, t)

                test_loss = self.__loop(self.test_loader, self.transunet.test_step, t)

            callback.epoch_end(epoch + 1,
                               {'loss': train_loss / len(self.train_loader),
                                'test_loss': test_loss / len(self.test_loader)})

            if callback.end_training:
                break
