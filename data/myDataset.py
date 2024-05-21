import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

from torchvision import transforms
import random


def build_samples(item, anchor_value, shots):
    positive = np.where(item >= anchor_value)[0]
    negtive = np.where(item < anchor_value)[0]

    np.random.shuffle(positive)
    np.random.shuffle(negtive)

    pos_sample = positive[:int((len(positive) * 0.5) // 1)]
    positive = positive[int((len(positive) * 0.5) // 1):]

    neg_sample = negtive[:int((len(negtive) * 0.5) // 1)]
    negtive = negtive[int((len(negtive) * 0.5) // 1):]

    sam_len = min(len(pos_sample), len(neg_sample))
    pr_len = min(len(positive), len(negtive))

    if sam_len == 0 or pr_len == 0:
        return -1

    if shots != 50 and sam_len < 10:
        return -1

    if shots == 50:
        pos_sample = pos_sample[:sam_len]
        neg_sample = neg_sample[:sam_len]
        positive = positive[:pr_len]
        negtive = negtive[:pr_len]
    else:
        pos_sample = pos_sample[:shots]
        neg_sample = neg_sample[:shots]
        positive = positive[:pr_len]
        negtive = negtive[:pr_len]

    grid_num = len(item)
    zero_pos_sample = torch.zeros(grid_num)
    zero_pos_sample[pos_sample] = 1
    zero_neg_sample = torch.zeros(grid_num)
    zero_neg_sample[neg_sample] = 1
    zero_postive = torch.zeros(grid_num)
    zero_postive[positive] = 1
    zero_negtive = torch.zeros(grid_num)
    zero_negtive[negtive] = 1

    return zero_pos_sample, zero_neg_sample, zero_postive, zero_negtive


class myDataset(Dataset):
    def __init__(self, gpu, shots, train=1):
        self.train = train
        # for training
        self.data = []
        self.anchor = []

        # for testing
        self.sample_pos = []
        self.sample_neg = []
        self.positive = []
        self.negative = []

        self.gpu = gpu
        self.shots = shots

    def append(self, input):
        # print(self.train)
        item, anchor_value = input

        if self.train == 1:
            self.data.append(item)
            self.anchor.append(anchor_value)
        else:
            for j in range(10):
                samples = build_samples(item, anchor_value, self.shots)
                sample_pos, sample_neg, positive, negative = samples
                self.sample_pos.append(sample_pos)
                self.sample_neg.append(sample_neg)
                self.positive.append(positive)
                self.negative.append(negative)

    def __len__(self):
        if self.train == 1:
            return len(self.data)
        else:
            return len(self.sample_neg)

    def __getitem__(self, index):
        if self.train == 1:
            samples = build_samples(self.data[index], self.anchor[index], self.shots)
            sample_pos, sample_neg, positive, negative = samples
            return sample_pos.to(self.gpu), sample_neg.to(self.gpu), \
                positive.to(self.gpu), negative.to(self.gpu)
        else:
            return self.sample_pos[index].to(self.gpu), self.sample_neg[index].to(self.gpu), \
                self.positive[index].to(self.gpu), self.negative[index].to(self.gpu)


class myDatasetRegression(Dataset):
    def __init__(self, gpu, train=1):
        self.train = train
        # for training
        self.data = []
        # for testing
        self.mask = []

        self.gpu = gpu

    def append(self, input):
        if self.train == 1:
            item = input
            self.data.append(item)
        else:
            item, input_mask = input
            self.data.append(item)
            self.mask.append(input_mask)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.train == 1:
            item = self.data[index]
            while True:
                index = np.arange(len(item))
                np.random.shuffle(index)
                input_index = index[:int(0.7 * len(index))]
                if np.sum(item[input_index]) > 0:
                    break
            input_mask = torch.zeros(len(item))
            input_mask[input_index] = 1
            return torch.FloatTensor(item).to(self.gpu), input_mask.to(self.gpu)
        else:
            return self.data[index].to(self.gpu), self.mask[index].to(self.gpu)
