import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn import preprocessing

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

class Preprocess:
    def __init__(self, data, textLabel, targetLabel):
        self.data = data
        self.preprocessedText = data
        self.textLabel = textLabel
        self.targetLabel = targetLabel
        
        self.encodeLabel
        self.tokenizeWord()
        self.tokenizeWord()
        self.removeStopword()
        # self.lemmatize()
        # stemming()

        self.preprocessedText = self.data

    def encodeLabel(self):
        le = preprocessing.LabelEncoder()

        self.data[self.targetLabel] = le.fit_transform(self.data[self.targetLabel]) #Mengganti data alphabet menjadi numerikal
        self.data = self.data.drop(df_train.columns[0], axis=1)

    def tokenizeWord(self):
        wordTokens = []
        for text in self.data[self.textLabel]:
            words = nltk.tokenize.word_tokenize(text)
            wordTokens.append(words)
        self.data['wordTokens'] = wordTokens

    def removeStopword(self):
        list_stopwords = set(stopwords.words('indonesian'))
        token_without_stopwords = []
        for tokens in self.data['wordTokens']:
            stopwordRemove = [word for word in tokens if not word in list_stopwords]
            token_without_stopwords.append(stopwordRemove)
        self.data['tokenWithoutStopwords'] = token_without_stopwords

    def stemming(self):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        stemmedToken = []

        for list_tokens in self.data['tokenWithoutStopwords']:
            temp = []
            for tokens in list_tokens:
                temp.append(stemmer.stem(tokens))
            stemmedToken.append(temp)
        self.data["StemmedToken"] = stemmedToken

    def lemmatize(self):
        from nltk.stem.wordnet import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        
        lemmatizedToken = []
        
        for list_tokens in self.data['tokenWithoutStopwords']:
            temp = []
            for tokens in list_tokens:
                temp.append(lemmatizer.lemmatize(tokens))
            lemmatizedToken.append(temp)
        self.data["LemmatizedToken"] = lemmatizedToken

    def getPreprocessedTexts(self):
        return self.preprocessedText