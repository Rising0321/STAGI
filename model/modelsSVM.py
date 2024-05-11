import numpy as np
from sklearn.svm import OneClassSVM

from utils.utils import ClassificationMetrices


class modelsSVM:
    def __init__(self, config):
        datas = np.load(config['matrix_path'])
        datas = datas.reshape(-1, datas.shape[2])
        self.feature = datas.transpose(0, 1)

    def run(self, test_files):
        metrices = ClassificationMetrices()
        from tqdm import tqdm
        print("+1")
        for test_tuple in tqdm(test_files):
            sample_list, positive_list, negative_list = test_tuple
            for i in range(len(sample_list)):
                acc_pos = [0, 0]
                acc_neg = [0, 0]
                sample = sample_list[i].cpu().numpy()
                positive = positive_list[i].cpu().numpy()
                negative = negative_list[i].cpu().numpy()
                model = OneClassSVM()
                input = []

                for j in range(len(sample)):
                    if sample[j] == 1:
                        input.append(self.feature[j])

                model.fit(input)

                for j in range(len(positive)):
                    if positive[j] == 1:
                        now = self.feature[j].reshape(1, -1)
                        if model.predict(now) == 1:
                            acc_pos[0] += 1
                        acc_pos[1] += 1

                for j in range(len(negative)):
                    if negative[j] == 1:
                        now = self.feature[j].reshape(1, -1)
                        if model.predict(now) == -1:
                            acc_neg[0] += 1
                        acc_neg[1] += 1
                metrices.update((acc_pos, acc_neg))
        metrices = metrices.output("test", 0)
