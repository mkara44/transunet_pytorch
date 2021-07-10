import torch
import numpy as np


def thresh_func(mask, thresh=0.5):
    mask[mask >= thresh] = 1
    mask[mask < thresh] = 0

    return mask


def dice_loss(pred, target):
    pred = torch.sigmoid(pred)

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = torch.sum(pred * target)
    pred_sum = torch.sum(pred * pred)
    target_sum = torch.sum(target * target)

    return 1 - ((2. * intersection + 1e-5) / (pred_sum + target_sum + 1e-5))


class EpochCallback:
    end_training = False
    not_improved_epoch = 0
    monitor_value = np.inf

    def __init__(self, model_name, total_epoch_num, model, optimizer, monitor=None, patience=None):
        if isinstance(model_name, str):
            model_name = [model_name]
            model = [model]
            optimizer = [optimizer]

        self.model_name = model_name
        self.total_epoch_num = total_epoch_num
        self.monitor = monitor
        self.patience = patience
        self.model = model
        self.optimizer = optimizer

    def __save_model(self):
        for m_name, m, opt in zip(self.model_name, self.model, self.optimizer):
            torch.save({'model_state_dict': m.state_dict(),
                        'optimizer_state_dict': opt.state_dict()},
                       m_name)

            print(f'Model saved to {m_name}')

    def epoch_end(self, epoch_num, hash):
        epoch_end_str = f'Epoch {epoch_num}/{self.total_epoch_num} - '
        for name, value in hash.items():
            epoch_end_str += f'{name}: {round(value, 4)} '

        print(epoch_end_str)

        if self.monitor is None:
            self.__save_model()

        elif hash[self.monitor] < self.monitor_value:
            print(f'{self.monitor} decreased from {round(self.monitor_value, 4)} to {round(hash[self.monitor], 4)}')

            self.not_improved_epoch = 0
            self.monitor_value = hash[self.monitor]
            self.__save_model()
        else:
            print(f'{self.monitor} did not decrease from {round(self.monitor_value, 4)}, model did not save!')

            self.not_improved_epoch += 1
            if self.patience is not None and self.not_improved_epoch >= self.patience:
                print("Training was stopped by callback!")
                self.end_training = True
