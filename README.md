# STAGI

## Data Processing

### Manhattan Pretrain Dataset
1. Download all the Yellow Taxi Trip Data and Green Taxi Trip Data from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
2. Place the data in data/raw.
3. run step-0-combine_all.py to generate the dataset.

### Manhattan Finetune Dataset
1. Download the POI data from https://data.cityofnewyork.us/City-Government/Points-Of-Interest/rxuy-2muj as csv type.
2. Place the POI data and new york taxi zone file in data/rawPOI.
3. run process_task_regression.py to generate the dataset.

## Pretrain
python train.py --pretrain 1

## Finetune
python train.py --pretrain 0 --load where_you_save_the_model