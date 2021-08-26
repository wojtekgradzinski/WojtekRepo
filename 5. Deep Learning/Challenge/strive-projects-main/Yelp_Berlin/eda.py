import numpy as np
import pandas as pd

df = pd.read_csv('business1.csv')

print(df.head())

print(df.info())

print(df.ndim)

print(df.describe())

print(df.isnull().sum())

print(df.nunique())