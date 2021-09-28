from word_embedding import WordEmbbeding
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import *
from shallow import *
from word_embedding import *

# df_train = pd.read_csv('./data/train.csv')
# df_test = pd.read_csv('./data/test.csv')

# preprocessTrain = Preprocess(df_train, 'text_a', 'label', 'train')
# preprocessTest = Preprocess(df_test, 'text_a', 'label', 'test')
# print(preprocessTrain.data.head())

print("METODE:")
print("Shallow Learning (1)")
print("Word Embedding (2)")
print("=========================")
print("Masukkan input sesuai nomor metode yang diinginkan")
method = int(input())

preprocessTrain = pd.read_csv('df_train_preprocessed.csv')
preprocessTest = pd.read_csv('df_test_preprocessed.csv')

if(method == 1):
    shallow = Shallow(preprocessTrain, preprocessTest)
    shallow.getMetrics()
elif(method == 2):
    wordEmbbeding = WordEmbbeding(preprocessTrain, preprocessTest)
    wordEmbbeding.getMetric()
