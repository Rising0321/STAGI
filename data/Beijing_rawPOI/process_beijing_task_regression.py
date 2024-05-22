import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from tqdm import tqdm

MID_TYPE_THRESHOLD = 10000
SAMPLE_SIZE = 1000
# 读取数据，筛选某类型数量超过一定值的类别，并随机采样一部分

with open('Beijing_raw_POI.csv', 'r', encoding='utf-8') as f:
    raw_data = pd.read_csv(f, usecols=['location_X', 'location_Y', 'big_type', 'mid_type'])

mid_type_counts = raw_data['mid_type'].value_counts()
a = len(mid_type_counts[mid_type_counts > MID_TYPE_THRESHOLD])

print(a)

selected_type = mid_type_counts[mid_type_counts > MID_TYPE_THRESHOLD].index
print(selected_type)
my_selected_type = ['交通地名']
filtered_data = raw_data[raw_data['mid_type'].isin(selected_type) & ~ raw_data['mid_type'].isin(my_selected_type)]
print('filtered_data:', len(filtered_data))



def transfer_gps_to_int(gps):
    return int((gps[0] - 116.2075) * 100) * 30 + int((gps[1] - 39.7523) * 100)


def is_legal(gps):
    return 116.2075 <= gps[0] < 116.5475 and 39.7523 <= gps[1] < 40.0423


res_tables = {}
progress_bar = tqdm(total=len(filtered_data), desc='isin_region')
for i, row in filtered_data.iterrows():
    lon, lat = row['location_X'], row['location_Y']
    big_type, mid_type = row['big_type'], row['mid_type']

    # 更新进度条
    progress_bar.update(1)

    if is_legal((lon, lat)):
        j = transfer_gps_to_int((lon, lat))
        cur_pair = (big_type, mid_type)
        if cur_pair not in res_tables:
            res_tables[cur_pair] = np.zeros(1020)
        res_tables[cur_pair][j] += 1

progress_bar.close()

for key in tqdm(res_tables, desc='save_npy'):
    name = str(key[0]) + "-" + str(key[1])
    if name == "购物服务-便民商店/便利店":
        name = "购物服务-便民商店"
    print(len(res_tables[key]))
    print(res_tables[key])
    np.save("../Beijing_regression_data/" + name, res_tables[key])
