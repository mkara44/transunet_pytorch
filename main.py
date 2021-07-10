import torch

# Additional Scripts
from train import TrainTestPipe


def main_pipeline():
    device = 'cpu:0'
    if torch.cuda.is_available():
        device = 'cuda:0'

    ttp = TrainTestPipe(train_path='/train',
                        test_path='/test',
                        device=device)

    ttp.train()


if __name__ == '__main__':
    main_pipeline()
