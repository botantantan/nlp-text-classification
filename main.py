import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import *

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

preprocess = Preprocess(df_train, 'text_a', 'label')
print(preprocess.data.head())
