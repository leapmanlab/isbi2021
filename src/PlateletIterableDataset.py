"""Platelet dataset that uses generators to stream sample windows from a large
data volume

"""
import random

import numpy as np
import torch

from torch.utils.data import IterableDataset


class ShuffleDataset(IterableDataset):
    """https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/6

    """
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass

class PlateletIterableDataset(IterableDataset):
    """Create a generator to yield platelet dataset samples

    """
    def __init__(
            self,
            images,
            labels=None,
            weight=None,
            train=True,
            min_nonzero_sample_ratio=0.05):
        """Constructor.

        Args:
            images: A generator yielding windows from some image data.
            labels: If supplied, a generator yielding windows from some label
                data. If `train==True`, labels must be supplied.
            weight: If supplied, a generator yielding windows from some weight
                array data.
            train:
            min_nonzero_sample_ratio:
        """
        super().__init__()
        self.images = images
        self.labels = labels
        self.weight = weight
        self.train = train
        self.min_nonzero_sample_ratio = min_nonzero_sample_ratio

        if self.train:
            assert not (labels is None)

        if self.train:
            if self.weight is None:
                self.data_iterator = iter(zip(self.images, self.labels))
            else:
                self.data_iterator = iter(zip(self.images, self.labels, self.weight))
        else:
            if self.labels is None:
                self.data_iterator = self.images
            else:
                self.data_iterator = iter(zip(self.images, self.labels))
        pass

    def __iter__(self):
        if self.train:
            for sample in self.data_iterator:
                nonzero_ratio = sample[1].mean()
                if nonzero_ratio >= self.min_nonzero_sample_ratio:
                    yield sample
                # while nonzero_ratio < self.min_nonzero_sample_ratio:
                #     sample = next(self.data_iterator)
                #     nonzero_ratio = sample[1].sum() / sample[1].size
                # yield sample
        else:
            yield from self.data_iterator

    def transform(self, image, label=None, weight=None):
        for i in [1, 2]:
            # Random flipping
            if random.random() > 0.5:
                image = np.flip(image, i).copy()
                if label is not None:
                    label = np.flip(label, i).copy()
                if weight is not None:
                    weight = np.flip(weight, i).copy()
        # Transform to tensor
        image = torch.from_numpy(image)
        if label is not None:
            label = torch.from_numpy(label)
        if weight is not None:
            weight = torch.from_numpy(weight)
        if label is None and weight is None:
            return image
        elif weight is None:
            return image, label
        else:
            return image, label, weight
