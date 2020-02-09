from sklearn import preprocessing
import pandas as pd

train_csv = pd.read_csv('../input/train_data.csv')

train_csv.dropna(axis=0, inplace=True)
train_csv.drop(['Purchased'], axis=1, inplace=True)
print(train_csv)
batch1 = pd.get_dummies(train_csv, prefix='city', drop_first=True)
print(batch1)