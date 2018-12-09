import numpy as np


class BaseDataset:
    def __init__(self, params, random_seed=1211):
        self.params = params
        self.random_state = np.random.RandomState(random_seed)
        self.epoch = 1

        self.batch_size = params['batch_size']

    def build(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def batch_generator(self):
        samples = []
        while True:
            for i, data in enumerate(self):
                sample = self.make_sample(data)
                if sample is not None:
                    samples.append(sample)
                if len(samples) == self.batch_size or (i == len(self) - 1 and samples):
                    yield self.make_batch(samples)
                    samples = []
            self.epoch += 1

    def make_sample(self, data):
        raise NotImplementedError

    def make_batch(self, samples):
        raise NotImplementedError
