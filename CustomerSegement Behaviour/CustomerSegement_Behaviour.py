import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

import warnings 
warnings.filterwarnings('ignore') 

df = pd.read_csv('data.csv')
#df['Accepted'] = df['Accepted'].str.replace('Accepted', '')
#print(df.head())



for col in df.columns:
    temp = df[col].isnull().sum()
    if temp>0:
        print(f'Column {col} has {temp} null values')
 


df = df.dropna()
print("Total missing values are:", len(df))
print(df.nunique())
parts = df["Dt_Customer"].str.split("-", n=3, expand=True)
df["day"] = parts[0].astype('int')
df["month"] = parts[1].astype('int')
df["year"] = parts[2].astype('int')

df.drop(['Z_CostContact', 'Z_Revenue', 'Dt_Customer'],
       axis=1, 
       inplace=True)


floats, objects = [], []
for col in df.columns:
    if df[col].dtype == object:
        objects.append(col)
    elif df[col].dtype == float:
        floats.append(col)

print(objects)
print(floats)

plt.subplots(figsize=(15,10))
for i, col in enumerate(objects):
    plt.subplot(2, 2, i + 1)
    df_melted = df.melt(id_vars=[col], value_vars=['Response'], var_name ='hue')
    sb.countplot(x=col, hue='value', data=df_melted)
plt.show()

print(df['Marital_Status'].value_counts())

