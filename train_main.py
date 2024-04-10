import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model.modelsAGI import ur_vit_base_patch16
from utils.utils import load_config, load_files, build_train_data, evaluation, init_seed, init_logging, output


def train(model, train_data, optimizer, scheduler, epoch, phase):
    sum_pos = (0, 0)
    sum_neg = (0, 0)

    model.train()
    for train_tuple in tqdm(train_data):
        optimizer.zero_grad()

        loss, acc_pos, acc_neg = model(train_tuple)
        # print(model.token_emb.word_embeddings.weight)
        loss.backward()
        optimizer.step()
        # set tqdm to show the loss
        # tqdm.write(f"Epoch: {epoch}, Loss: {loss.item()}, Acc Pos: {acc_pos}, Acc Neg: {acc_neg}")
        sum_pos = (sum_pos[0] + acc_pos[0], sum_pos[1] + acc_pos[1])
        sum_neg = (sum_neg[0] + acc_neg[0], sum_neg[1] + acc_neg[1])

    scheduler.step()
    roc_auc, f1, accuracy, precision, recall = evaluation(sum_pos, sum_neg)
    output(f"{phase}: Epoch: {epoch}, Acc Pos: {sum_pos[0], sum_pos[1]}, Acc Neg: {sum_neg[0], sum_neg[1]}")
    output(
        f"roc_auc: {roc_auc}, f1: {f1}, accuracy: {accuracy}, precision: {precision}, recall: {recall}")


def test(model, test_data, epoch, phase):
    sum_pos = (0, 0)
    sum_neg = (0, 0)
    model.eval()
    with torch.no_grad():
        for eval_tuple in tqdm(test_data):
            loss, acc_pos, acc_neg = model(eval_tuple)
            sum_pos = (sum_pos[0] + acc_pos[0], sum_pos[1] + acc_pos[1])
            sum_neg = (sum_neg[0] + acc_neg[0], sum_neg[1] + acc_neg[1])
        roc_auc, f1, accuracy, precision, recall = evaluation(sum_pos, sum_neg)
        output(f"{phase}: Epoch: {epoch}, Acc Pos: {sum_pos[0], sum_pos[1]}, Acc Neg: {sum_neg[0], sum_neg[1]}")
        output(
            f"roc_auc: {roc_auc}, f1: {f1}, accuracy: {accuracy}, precision: {precision}, recall: {recall}")
    return roc_auc, f1, accuracy, precision, recall


def main(args):
    init_seed(args.seed)

    init_logging(args)

    config = load_config(args.data)

    files = load_files(config['file_path'])

    train_data, val_data, test_data = build_train_data(files, args.batch_size, args.gpu)

    model = ur_vit_base_patch16(config['N']).to(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_eval_metrices = -1
    best_test_metrices = -1
    cnt = 0

    for epoch in range(1, args.epochs):
        train(model, train_data, optimizer, scheduler, epoch, "train")

        if epoch % 10 == 0:

            eval_metrices = test(model, val_data, epoch, "val")

            test_metrices = test(model, test_data, epoch, "test")

            if best_eval_metrices == -1 or eval_metrices[0] > best_eval_metrices[0]:
                best_eval_metrices = eval_metrices
                best_test_metrices = test_metrices
                cnt = 0
            else:
                cnt += 1
                if cnt == args.patience:
                    output("Early stopping")
                    output(f"Best Metrices: {best_test_metrices}")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        type=str,
                        help='Dataset name (eg. Manhattan, Beijing)',
                        default='Beijing')

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
                        default="cuda:3")

    parser.add_argument('--seed',
                        type=int,
                        help='Random seed',
                        default=42)

    parser.add_argument('--patience',
                        type=int,
                        help='Patience',
                        default=20)

    parser.add_argument('--model_name',
                        type=str,
                        help='name of the model',
                        default="test")

    args = parser.parse_args()

    main(args)
