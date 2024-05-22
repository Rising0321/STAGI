import numpy as np
from sklearn.svm import OneClassSVM, SVC

from utils.utils import ClassificationMetrices

import torch
import torch.nn as nn


def combine_datas(config):
    walk_path = config['file_path']
    import os
    res = []
    for root, nowdir, nowfiles in os.walk(walk_path):
        for file in nowfiles:
            if file != "flow_matrix.npy":
                res.append(root + "/" + file)

    res = res[:int(len(res) * 0.7)]
    datas = []
    for item in res:
        now = np.load(item)
        datas.append(now)

    datas = np.array(datas)
    datas = datas.transpose()
    print(datas.shape)
    return datas


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def fit(self, x, label):
        x = torch.tensor(x).float()
        label = torch.tensor(label)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(100):
            optimizer.zero_grad()
            output = self(x)
            # print(output)
            # print(label)
            loss = criterion(output, label.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

    def predict(self, x):
        x = torch.tensor(x).float()
        output = self(x)
        return 1 if output > 0.5 else 0


class modelsMLP:
    def __init__(self, config):
        '''
        datas = np.load(config['matrix_path'])
        datas = datas.reshape(-1, datas.shape[2])
        self.feature = datas.transpose()
        print(self.feature.shape)
        '''
        self.feature = combine_datas(config)
        # print(self.feature.shape)

    def run(self, test_files):
        metrices = ClassificationMetrices()
        from tqdm import tqdm
        print("+1")
        cnt = 0
        import time
        now_time = time.time()
        for test_tuple in tqdm(test_files):
            sample_pos_list, sample_neg_list, positive_list, negative_list = test_tuple
            for i in range(len(sample_pos_list)):
                acc_pos = [0, 0]
                acc_neg = [0, 0]
                sample_pos = sample_pos_list[i].cpu().numpy()
                sample_neg = sample_neg_list[i].cpu().numpy()
                positive = positive_list[i].cpu().numpy()
                negative = negative_list[i].cpu().numpy()

                model = MLP(len(self.feature[0]), len(self.feature[0]) * 4, 1)

                input = []
                label = []
                for j in range(len(sample_pos)):
                    if sample_pos[j] == 1:
                        input.append(self.feature[j])
                        label.append(1)
                    if sample_neg[j] == 1:
                        input.append(self.feature[j])
                        label.append(0)

                model.fit(input, label)

                for j in range(len(positive)):
                    if positive[j] == 1:
                        now = self.feature[j].reshape(1, -1)
                        if model.predict(now) == 1:
                            acc_pos[0] += 1
                        acc_pos[1] += 1

                for j in range(len(negative)):
                    if negative[j] == 1:
                        now = self.feature[j].reshape(1, -1)
                        if model.predict(now) == 0:
                            acc_neg[0] += 1
                        acc_neg[1] += 1
                metrices.update((acc_pos, acc_neg))
            cnt += 1
            if cnt % 100 == 0:
                print(metrices.output("test", cnt))
        metrices = metrices.output("test", 0)
        print(time.time() - now_time)
