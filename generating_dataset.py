# Imports
import pandas as pd

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# Get the data
df_train = pd.read_csv("dataset/train_dataset_train.csv")
df_test = pd.read_csv("dataset/test_dataset_test.csv")

m = {"Y": 1, "N": 0}
df_train["is_privatecategory"] = df_train["is_privatecategory"].map(m)
df_test["is_privatecategory"] = df_test["is_privatecategory"].map(m)
df_train["is_in_yandex"] = df_train["is_in_yandex"].map(m)
df_test["is_in_yandex"] = df_test["is_in_yandex"].map(m)
df_train["is_return"] = df_train["is_return"].map(m)
df_test["is_return"] = df_test["is_return"].map(m)

df_train["oper_type"] = df_train["oper_type + oper_attr"].apply(lambda x: x.split('_')[0])
df_train["oper_attr"] = df_train["oper_type + oper_attr"].apply(lambda x: x.split('_')[1])
df_test["oper_type"] = df_test["oper_type + oper_attr"].apply(lambda x: x.split('_')[0])
df_test["oper_attr"] = df_test["oper_type + oper_attr"].apply(lambda x: x.split('_')[1])

categorical = [
    "type",
    "priority",
    "class",
    "mailtype",
    "mailctg",
    "directctg",
    "postmark",
    "is_wrong_sndr_name",
    "is_wrong_rcpn_name",
    "is_wrong_phone_number",
    "is_wrong_address",
]
for c in tqdm(categorical):
    m = {}
    for i, u in enumerate(set(df_train[c].unique()) or set(df_test[c].unique())):
        m[u] = str(i)
    df_train[c] = df_train[c].map(m).astype(str)
    df_test[c] = df_test[c].map(m).astype(str)


df_train.to_csv('dataset/train.csv', index=False)
df_test.to_csv('dataset/test.csv', index=False)

# Check dataset
print(df_train.shape)
print(df_train.columns)
print(df_train.head())
print()
print(df_test.shape)
print(df_test.columns)
print(df_test.head())