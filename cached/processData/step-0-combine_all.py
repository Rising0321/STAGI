import math
import os
import numpy as np
import pandas as pd
import csv
import re
import torch

raw_taxi_data_dir = '../raw/'

month_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def to_date(now):
    mat = re.search("(\d{4})-(\d{2})-(\d{2})\ (\d{2}):(\d{2})", now)
    return int(mat.group(1)), int(mat.group(2)), int(mat.group(3)), int(mat.group(4))


def to_week(date):
    st_day = 4
    year, month, day, hour = to_date(date)
    temp = 0
    for i in range(month - 1):
        temp += month_days[i]
    temp += day - 1 + st_day
    return temp % 7


def build_odMatrix(now):
    od_matrix = torch.zeros([63, 63])
    for i in range(len(now)):
        od_matrix[now[i][0]][now[i][1]] += 1
    return od_matrix


def construct_taxi_sequence():
    print("Begin Taxi Constructing.")

    data_list = []
    for name in os.listdir(raw_taxi_data_dir):

        print("Begin Constructing {}.".format(name))

        if name.startswith('yellow'):
            # with open((os.path.join(raw_taxi_data_dir, name)), 'rb') as f:
            f = pd.read_parquet(os.path.join(raw_taxi_data_dir, name))
            for row in f.values:
                if to_date(str(row[1]))[0] == 2016:
                    data_list.append([str(row[1]), int(row[7]), int(row[8])])

        elif name.startswith('green'):
            f = pd.read_parquet(os.path.join(raw_taxi_data_dir, name))
            for row in f.values:
                if to_date(str(row[1]))[0] == 2016:
                    data_list.append([str(row[1]), int(row[5]), int(row[6])])
        else:
            continue
        print("Constructing {} finished.".format(name))

    legal_id = [4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100, 107, 113, 114, 116, 120, 125,
                127, 128, 137, 140, 141, 142, 143, 144, 148, 151, 152, 158, 161, 162, 163, 164, 166, 170, 186, 209, 211,
                224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263]

    mydic = {}
    for i in range(len(legal_id)):
        mydic[legal_id[i]] = i

    # build a zero matrix which is 63*63 *(7*24)

    od_matrix = np.zeros([63, 63, 7, 24])
    cnt_day = np.zeros([7])
    st_day = 4

    for i in range(366):
        cnt_day[st_day] += 1
        st_day = (st_day + 1) % 7

    for item in data_list:
        if item[1] in legal_id and item[2] in legal_id:
            item[1] = mydic[item[1]]
            item[2] = mydic[item[2]]

            st = item[1]
            ed = item[2]

            week = to_week(item[0])
            hour = to_date(item[0])[3]
            od_matrix[st][ed][week][hour] += 1 / cnt_day[week]

    np.save("../graph/od_matrix.npy", od_matrix)


if __name__ == '__main__':
    construct_taxi_sequence()
