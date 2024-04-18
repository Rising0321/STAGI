import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

from torchvision import transforms


class myDataset(Dataset):
    def __init__(self, gpu, train=1):
        self.train = train
        # for training
        self.data = []
        self.anchor = []
        # for testing
        self.sample = []
        self.positive = []
        self.negative = []

        self.gpu = gpu

    def append(self, input):
        if self.train == 1 or self.train == 2:
            item, anchor_value = input
            self.data.append(item)
            self.anchor.append(anchor_value)
        else:
            item, sample, postive, negative = input
            self.data.append(item)
            self.sample.append(sample)
            self.positive.append(postive)
            self.negative.append(negative)
    def __len__(self):
        if self.train == 1 or self.train == 2:
            return len(self.data)
        else:
            return len(self.sample)

    def __getitem__(self, index):
        if self.train == 1:
            item = self.data[index]
            anchor_value = self.anchor[index]

            positive = np.where(item >= anchor_value)[0]
            np.random.shuffle(positive)
            sample = positive[:int((len(positive) * 0.7) // 1)]
            positive = positive[int((len(positive) * 0.7) // 1):]

            negtive = np.where(item < anchor_value)[0]
            np.random.shuffle(negtive)
            negtive = negtive[:min(len(negtive), len(positive))]

            grid_num = len(item)
            zero_sample = torch.zeros(grid_num)
            zero_sample[sample] = 1
            zero_postive = torch.zeros(grid_num)
            zero_postive[positive] = 1
            zero_negtive = torch.zeros(grid_num)
            zero_negtive[negtive] = 1
            return zero_sample.to(self.gpu), zero_postive.to(self.gpu), zero_negtive.to(self.gpu)
        elif self.train == 2:
            item = self.data[index]
            anchor_value = self.anchor[index]

            positive = np.where(item >= anchor_value)[0]
            np.random.shuffle(positive)  # 是不是不用打乱了?
            negtive = np.where(item < anchor_value)[0]
            np.random.shuffle(negtive)
            # negtive = negtive[:min(len(negtive), len(positive))] # 这应该不用裁剪

            grid_num = len(item)
            zero_sample = torch.zeros(grid_num)
            zero_sample[positive] = 1
            zero_postive = torch.zeros(grid_num)
            zero_postive[positive] = 1
            zero_negtive = torch.zeros(grid_num)
            zero_negtive[negtive] = 1
            # print(torch.allclose(zero_sample,zero_postive))
            return zero_sample.to(self.gpu), zero_postive.to(self.gpu), zero_negtive.to(self.gpu)

        else:
            return self.sample[index].to(self.gpu), self.positive[index].to(self.gpu), \
                    self.negative[index].to(self.gpu)

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
                index = np.arrange(len(item))
                np.random.shuffle(index)
                input_index = index[:(0.7 * len(index)) // 1]
                if np.sum(item[input_index]) > 0:
                    break
            input_mask = torch.zeros(item)
            input_mask[input_index] = 1
            return item.to(self.gpu), input_mask.to(self.gpu)
        else:
            return self.data[index].to(self.gpu), self.mask[index].to(self.gpu)
