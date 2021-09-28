import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import *
from shallow import *

# df_train = pd.read_csv('./data/train.csv')
# df_test = pd.read_csv('./data/test.csv')

# preprocessTrain = Preprocess(df_train, 'text_a', 'label', 'train')
# preprocessTest = Preprocess(df_test, 'text_a', 'label', 'test')
# print(preprocessTrain.data.head())

preprocessTrain = pd.read_csv('df_train_preprocessed.csv')
preprocessTest = pd.read_csv('df_test_preprocessed.csv')
shallow = Shallow(preprocessTrain, preprocessTest)
