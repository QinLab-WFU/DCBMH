from argparse import Namespace

import torch


class XBM:
    def __init__(self, batchsize, n_bits, n_classes, n_samples):
        self.batch_size = batchsize
        self.n_bits = n_bits
        self.n_classes = n_classes
        self.max_size = min(2048, n_samples)

        assert self.max_size >= self.batch_size
        assert self.max_size % self.batch_size == 0

        self.feats = torch.zeros(self.max_size, self.n_bits, device='cuda:0')
        self.targets = torch.zeros(self.max_size, self.n_classes, device='cuda:0')
        self.ptr = 0

    def get(self):
        if self.ptr == self.max_size:
            return self.feats, self.targets
        else:
            return self.feats[: self.ptr], self.targets[: self.ptr]

    def set(self, feats, targets):
        # print(len(targets))
        # print(targets)
        assert self.batch_size == len(targets)

        self.ptr += self.batch_size
        if self.ptr > self.max_size:
            self.ptr = self.max_size
            self.feats = torch.roll(self.feats, shifts=-self.batch_size, dims=0)
            self.targets = torch.roll(self.targets, shifts=-self.batch_size, dims=0)

        self.feats[-self.batch_size :] = feats
        self.targets[-self.batch_size :] = targets
