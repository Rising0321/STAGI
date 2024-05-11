import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model.modelsAGI import ur_vit_base_patch16
from model.modelsSVM import modelsSVM
from utils.utils import load_config, load_files, build_train_data, init_seed, init_logging, output, evaluation
from utils.utils import ClassificationMetrices, RegressionMetrices


def train(model, train_data, optimizer, scheduler, epoch, regression, phase):
    if regression:
        metrices = RegressionMetrices()
    else:
        metrices = ClassificationMetrices()

    model.train()
    for train_tuple in tqdm(train_data):
        optimizer.zero_grad()

        loss, acc = model(train_tuple)
        loss.backward()
        optimizer.step()

        metrices.update(acc)

    scheduler.step()
    metrices = metrices.output(phase, epoch)

    return metrices


def test(model, test_data, epoch, regression, phase):
    if regression:
        metrices = RegressionMetrices()
    else:
        metrices = ClassificationMetrices()

    model.eval()
    with torch.no_grad():
        for eval_tuple in tqdm(test_data):
            loss, acc = model(eval_tuple)

            metrices.update(acc)

    metrices = metrices.output(phase, epoch)

    return metrices


def baseline(baseline_data, test_data, N):
    cnt_one = torch.zeros(N)
    cnt_zero = torch.zeros(N)
    for i, data in enumerate(baseline_data):
        # data shape is (3,32,1020)
        pos_idx = np.where(data[1].cpu() == 1)[1]
        neg_idx = np.where(data[2].cpu() == 1)[1]
        for idx in pos_idx:
            cnt_one[idx] += 1
        for idx in neg_idx:
            cnt_zero[idx] += 1

    print(cnt_one)
    print(cnt_zero)

    result = cnt_one > cnt_zero
    result_cnt_one = torch.where(result * cnt_one != 0, torch.tensor(1), torch.tensor(0)).cuda()

    print(torch.sum(result_cnt_one))
    sum_pos = (0, 0)
    sum_neg = (0, 0)
    for i, eval_tuple in enumerate(test_data):
        sample = eval_tuple[0]  # shape [64 x 1020]
        positive = eval_tuple[1].cuda()
        negative = eval_tuple[2].cuda()

        acc_pos = (int(torch.sum(result_cnt_one * positive)), int(torch.sum(positive)))
        acc_neg = (int(torch.sum((1 - result_cnt_one) * negative)), int(torch.sum(negative)))

        sum_pos = (sum_pos[0] + acc_pos[0], sum_pos[1] + acc_pos[1])
        sum_neg = (sum_neg[0] + acc_neg[0], sum_neg[1] + acc_neg[1])

    roc_auc, f1, accuracy, precision, recall = evaluation(sum_pos, sum_neg)
    output(f"test: Acc Pos: {sum_pos[0], sum_pos[1]}, Acc Neg: {sum_neg[0], sum_neg[1]}")
    output(
        f"roc_auc: {roc_auc}, f1: {f1}, accuracy: {accuracy}, precision: {precision}, recall: {recall}")


def main(args):
    init_seed(args.seed)

    init_logging(args)

    config = load_config(args.data)

    files = load_files(args.pretrain, config)

    train_data, val_data, test_data, baseline_data = build_train_data(files, args.batch_size, args.gpu, args.regression)

    if args.model == "baseline":
        baseline(baseline_data, test_data, config['N'])
        exit(0)
    elif args.model == "SVM":
        model = modelsSVM(config)
        model.run(test_data)
        exit(0)

    model = ur_vit_base_patch16(config['N'], args.regression).to(args.gpu)

    if args.load_path != "":
        model.load_state_dict(torch.load(args.load_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_eval_metrices = -1
    best_test_metrices = -1
    cnt = 0

    for epoch in range(1, args.epochs):
        train(model, train_data, optimizer, scheduler, epoch, args.regression, "train")

        eval_metrices = test(model, val_data, epoch, args.regression, "val")

        test_metrices = test(model, test_data, epoch, args.regression, "test")

        if best_eval_metrices == -1 or eval_metrices[0] > best_eval_metrices[0]:
            best_eval_metrices = eval_metrices
            best_test_metrices = test_metrices
            cnt = 0
            torch.save(model.state_dict(), f"./checkpoints/{args.data}/{args.model}.pth")
        else:
            cnt += 1
            if cnt == args.patience:
                output("Early stopping")
                output(f"Best Metrices: {best_test_metrices}")
                break

        if epoch % args.test_epochs == 0:
            torch.save(model.state_dict(), f"./checkpoints/{args.data}/{args.model}-{epoch}.pth")

    # save the model at "./checkpoints/args.data/model_name-final.pth"
    torch.save(model.state_dict(), f"./checkpoints/{args.data}/{args.model}-final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        type=str,
                        help='Dataset name (eg. Manhattan, Beijing)',
                        default='Manhattan')

    parser.add_argument('--batch_size',
                        type=int,
                        help='batch_size',
                        default=16)

    parser.add_argument('--lr',
                        type=float,
                        help='Learning Rate',
                        default=1e-4)

    parser.add_argument('--epochs',
                        type=int,
                        help='Number of epochs',
                        default=10000000)

    parser.add_argument('--gpu',
                        type=str,
                        help='GPU',
                        default="cuda:0")

    parser.add_argument('--seed',
                        type=int,
                        help='Random seed',
                        default=42)

    parser.add_argument('--patience',
                        type=int,
                        help='Patience',
                        default=20)

    parser.add_argument('--model',
                        type=str,
                        help='model type',
                        default="SVM")

    parser.add_argument('--model_name',
                        type=str,
                        help='name of the model',
                        default="test")

    parser.add_argument('--load_path',
                        type=str,
                        help='load_path',
                        default="/home/zhangrx/STAGI/checkpoints/Manhattan/agi.pth")

    parser.add_argument('--regression',
                        type=int,
                        help='regression or classification',
                        default=0)

    parser.add_argument('--test_epochs',
                        type=int,
                        help='test_epochs',
                        default=10)

    parser.add_argument('--save_epochs',
                        type=int,
                        help='save_epochs',
                        default=10)

    parser.add_argument('--pretrain',
                        type=int,
                        help='pretrain',
                        default=0)

    args = parser.parse_args()

    main(args)
