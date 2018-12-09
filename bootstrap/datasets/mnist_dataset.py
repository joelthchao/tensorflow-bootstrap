from keras.datasets.mnist import load_data
import numpy as np

from bootstrap.datasets.base_dataset import BaseDataset


MNIST_CLASS_NUM = 10


class MnistDataset(BaseDataset):
    def __init__(self, params, random_seed=1211, mode='train'):
        super().__init__(params, random_seed)
        self.mode = mode
        self.num_class = params['num_class']

    def build(self):
        (x_train, y_train), (x_test, y_test) = load_data()
        if self.mode == 'train':
            self.x = x_train
            self.y = y_train
        elif self.mode == 'test':
            self.x = x_test
            self.y = y_test
        else:
            raise Exception
        print('Load dataset. shape={}'.format(x_train.shape))
        self.idx = 0

    def __next__(self):
        data = self.x[self.idx], self.y[self.idx]
        self.idx += 1
        if self.idx >= len(self):
            self.idx = 0
            self.epoch += 1
        return data

    def __len__(self):
        return len(self.x)

    def make_sample(self, data):
        x_, y_ = data
        x = x_.flatten()
        y = np.zeros(self.num_class)
        y[y_] = 1
        return x, y

    def make_batch(self, samples):
        batch_x, batch_y = [], []
        for x, y in samples:
            batch_x.append(x)
            batch_y.append(y)
        return {
            'x': np.array(batch_x),
            'y': np.array(batch_y),
        }
