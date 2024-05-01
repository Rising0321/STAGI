import os
import numpy as np
import pandas as pd
import re
import torch

raw_taxi_data_dir = './'

month_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

with open('../rawPOI/taxi_zones.csv', 'r') as f:
    csv = pd.read_csv(f)

legal_id = []
# build the MULTIPOLYGON from the csv's row the_geom
# iterate over the rows of the csv
for i, row in csv.iterrows():
    # get the_geom from the row
    # print(row['borough'])
    if row['borough'] == "Manhattan":
        legal_id.append(row['OBJECTID'])


def to_date(now):
    mat = re.search("(\d{4})-(\d{2})-(\d{2})\ (\d{2}):(\d{2})", now)
    return int(mat.group(1)), int(mat.group(2)), int(mat.group(3)), int(mat.group(4))


def to_day(date):
    year, month, day, hour = to_date(date)
    temp = 0
    for i in range(month - 1):
        temp += month_days[i]
    temp += day - 1
    return temp


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

    mydic = {}
    for i in range(len(legal_id)):
        mydic[legal_id[i]] = i

    # build a zero matrix which is 63*63 *(7*24)

    flow_matrix = np.zeros([366, 24, len(legal_id)])

    for item in data_list:
        if item[1] in legal_id:
            item[1] = mydic[item[1]]

            st = item[1]

            day = to_day(item[0])
            hour = to_date(item[0])[3]
            flow_matrix[day][hour][st] += 1

    np.save("../regression_data/od_matrix.flow_matrix", flow_matrix)


if __name__ == '__main__':
    construct_taxi_sequence()
