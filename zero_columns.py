import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

# CHO data frame
names = ['smiles', 'Tm', 'Tg']
for i in range(200):
    names.append(str(i))
df_cho = pd.read_csv('median_rdd_CHO_new.txt', delim_whitespace=True, names=names)
df_cho = df_cho.drop(['smiles'], axis=1)
names.insert(0, 'name')
names.remove('smiles')
names.remove('name')

zero_cols_cho = []
for col in names:
    if np.mean(df_cho[col]) == 0:
        df_cho = df_cho.drop([col], axis=1)
        zero_cols_cho.append(int(col))


# print(zero_cols_cho)
# df_cho = np.array(df_cho)
# print(df_cho.shape)

# nhal data frame
names = ['smiles', 'Tm', 'Tg']
for i in range(200):
    names.append(str(i))
df_nhal = pd.read_csv('median_rdd_nhal.txt', delim_whitespace=True, names=names)
df_nhal = df_nhal.drop(['smiles'], axis=1)
names.insert(0, 'name')
names.remove('smiles')
names.remove('name')

zero_cols_nhal = []
for col in names:
    if np.mean(df_nhal[col]) == 0:
        df_nhal = df_nhal.drop([col], axis=1)
        zero_cols_nhal.append(int(col))

# print(zero_cols_nhal)
# df_nhal = np.array(df_nhal)
# print(df_nhal.shape)
