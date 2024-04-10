import json
import numpy as np
import torch

from data.myDataset import myDataset

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

import logging
import os


def load_config(dataset):
    with open("./data/config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)

    config = config[dataset]

    return config


def load_files(walk_path):
    print(walk_path)
    import os
    res = []
    for root, nowdir, nowfiles in os.walk(walk_path):
        for file in nowfiles:
            res.append(root + "/" + file)

    datas = []
    for item in res:
        now = np.load(item)
        datas.append(now)

    return datas


def evaluation(sum_pos, sum_neg):
    # sum pos is in the format of (right, total), sum neg is in the format of (right, total)
    # please calculate the roc_auc, f1, accuracy, precision, recall using sklearn

    truth = [1] * sum_pos[1] + [0] * sum_neg[1]
    prediction = [1] * sum_pos[0] + [0] * (sum_pos[1] - sum_pos[0]) + [0] * sum_neg[0] + [1] * (sum_neg[1] - sum_neg[0])

    roc_auc = roc_auc_score(truth, prediction)
    f1 = f1_score(truth, prediction)
    accuracy = accuracy_score(truth, prediction)
    precision = precision_score(truth, prediction)
    recall = recall_score(truth, prediction)

    return roc_auc, f1, accuracy, precision, recall


def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logging(args):
    logging.basicConfig(level=logging.INFO)
    log_dir = os.path.join("./log", args.data)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'{args.model_name}_{args.seed}.log')
    file_handler = logging.FileHandler(log_file)
    logging.getLogger().addHandler(file_handler)
    logging.info(args)


def output(now):
    # print(now)
    logging.info(now)


# sample - positive - negative
def build_data(data, files, bsz, gpu, train=1):
    res = myDataset(gpu, train)

    for item in data:
        # iterater rate from 0.2 to 1
        for rate in range(5, 95, 5):
            # chosse the top k the largest value from the numpy array and return the boolean array
            anchor_value = np.percentile(item, 100 - rate)

            if anchor_value == 0:
                continue

            if len(np.where(item >= anchor_value)[0]) <= 1:
                continue

            if train == 1:
                res.append((item, anchor_value))

            else:
                for j in range(10):
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

                    res.append((zero_sample, zero_postive, zero_negtive))

    return torch.utils.data.DataLoader(res, batch_size=bsz,
                                       shuffle=True)


def build_train_data(files, batch_size, gpu):
    print(len(files))
    rate_train, rate_val, rate_test = 0.7, 0.15, 0.15

    train_data = files[:int(len(files) * rate_train)]
    val_data = files[int(len(files) * rate_train):int(len(files) * (rate_train + rate_val))]
    test_data = files[int(len(files) * (rate_train + rate_val)):]

    train_set = build_data(train_data, files, batch_size, gpu, train=1)
    val_set = build_data(val_data, files, batch_size, gpu, train=0)
    test_set = build_data(test_data, files, batch_size, gpu, train=0)

    return train_set, val_set, test_set
