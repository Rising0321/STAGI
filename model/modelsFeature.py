import numpy as np
from sklearn.svm import OneClassSVM, SVC

from utils.utils import ClassificationMetrices


def combine_datas(config):
    walk_path = config['file_path']
    import os
    res = []
    for root, nowdir, nowfiles in os.walk(walk_path):
        for file in nowfiles:
            if file != "flow_matrix.npy":
                res.append(root + "/" + file)

    datas = []
    for item in res:
        now = np.load(item)
        datas.append(now)

    datas = np.array(datas).transpose(0, 1)
    print(datas.shape)
    return datas


def calc_output(feature):
    means = []
    for j in range(len(feature[0])):
        means += [np.mean(feature[:, j])]
    output = []
    for i in range(len(feature)):
        cnt = 0
        for j in range(len(feature[i])):
            if feature[i][j] > means[j]:
                cnt += 1
        if cnt > len(feature[i]) / 2:
            output.append(1)
        else:
            output.append(-1)
    return output


class modelsFeature:
    def __init__(self, config):
        self.feature = combine_datas(config)
        self.output = calc_output(self.feature)

    def run(self, test_files):
        metrices = ClassificationMetrices()
        from tqdm import tqdm
        print("+1")
        cnt = 0
        for test_tuple in tqdm(test_files):
            sample_pos_list, sample_neg_list, positive_list, negative_list = test_tuple
            for i in range(len(sample_pos_list)):
                acc_pos = [0, 0]
                acc_neg = [0, 0]
                positive = positive_list[i].cpu().numpy()
                negative = negative_list[i].cpu().numpy()

                for j in range(len(positive)):
                    if positive[j] == 1:
                        if self.output[j] == 1:
                            acc_pos[0] += 1
                        acc_pos[1] += 1

                for j in range(len(negative)):
                    if negative[j] == 1:
                        if self.output[j] == -1:
                            acc_neg[0] += 1
                        acc_neg[1] += 1
                metrices.update((acc_pos, acc_neg))
            cnt += 1
            if cnt % 100 == 0:
                print(metrices.output("test", cnt))
        metrices = metrices.output("test", 0)
