import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn import metrics
import tensorflow
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K

class WordEmbbeding:
  def __init__(self, dataTrain, dataTest) -> None:
    self.dataTrain = dataTrain
    self.dataTest = dataTest

    self.addCleanText()
    self.nGramDetection()
    self.w2vFitting()
    self.featureEngineering()
    self.getMatOfEmb()
<<<<<<< HEAD
    # self.attention_layer()
=======
>>>>>>> 4b6a184ec107bf5ad1926e081d863da6cce45f2f
    self.genModel()
    self.trainData()
    # self.getMetric()

  def addCleanText(self):
    text_clean = []
    for list_tokens in self.dataTrain['StemmedToken']:
      # print(list_tokens.replace("[","").replace("'","").replace(","," ").replace("]",""))
      temp = list_tokens.replace("[","").replace("'","").replace(","," ").replace("]","")
      # temp = ' '.join(tokens)
      text_clean.append(temp)
    self.dataTrain["text_clean"] = text_clean

    text_clean = []
    for list_tokens in self.dataTest['StemmedToken']:
      temp = list_tokens.replace("[","").replace("'","").replace(","," ").replace("]","")
      # temp = ' '.join(tokens)
      text_clean.append(temp)
    self.dataTest["text_clean"] = text_clean
  
  def nGramDetection(self):
    self.corpus = self.dataTrain["text_clean"]

    self.lst_corpus = []
    for string in self.corpus:
      lst_words = string.split()
      lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, len(lst_words), 1)]
      self.lst_corpus.append(lst_grams)
<<<<<<< HEAD
    
    delimiterVar = " ".encode(encoding='utf8')
=======

>>>>>>> 4b6a184ec107bf5ad1926e081d863da6cce45f2f
    self.bigrams_detector = gensim.models.phrases.Phrases(self.lst_corpus, min_count=5, threshold=10)
    self.bigrams_detector = gensim.models.phrases.Phraser(self.bigrams_detector)
    self.trigrams_detector = gensim.models.phrases.Phrases(self.bigrams_detector[self.lst_corpus], min_count=5, threshold=10)
    self.trigrams_detector = gensim.models.phrases.Phraser(self.trigrams_detector)
  
  def w2vFitting(self):
    self.nlp = gensim.models.word2vec.Word2Vec(self.lst_corpus, vector_size=300, window=8, min_count=1, sg=1)

  def featureEngineering(self):
    # get x_train
    tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(self.lst_corpus)
    self.dic_vocabulary = tokenizer.word_index
    lst_text2seq= tokenizer.texts_to_sequences(self.lst_corpus)
    self.x_train = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=15, padding="post", truncating="post")

    # get x_test
    self.corpus = self.dataTest["text_clean"]
    self.lst_corpus = []
    for string in self.corpus:
      lst_words = string.split()
      lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, len(lst_words), 1)]
      self.lst_corpus.append(lst_grams)

    self.lst_corpus = list(self.bigrams_detector[self.lst_corpus])
    self.lst_corpus = list(self.trigrams_detector[self.lst_corpus])

    lst_text2seq = tokenizer.texts_to_sequences(self.lst_corpus)
    self.x_test = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=15, padding="post", truncating="post")


  def getMatOfEmb(self):
    self.embeddings = np.zeros((len(self.dic_vocabulary)+1, 300))
    for word,idx in self.dic_vocabulary.items():
      try:
          self.embeddings[idx] = self.nlp[word]
      except:
          pass

  def attention_layer(self, inputs, neurons):
    x = layers.Permute((2,1))(inputs)
    x = layers.Dense(neurons, activation="softmax")(x)
    x = layers.Permute((2,1), name="attention")(x)
    x = layers.multiply([inputs, x])
    return x
  
  def genModel(self):
    x_in = layers.Input(shape=(15,))
    x = layers.Embedding(input_dim=self.embeddings.shape[0], output_dim=self.embeddings.shape[1], weights=[self.embeddings], input_length=15, trainable=False)(x_in)
    x = self.attention_layer(x, neurons=15)
    x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2))(x)
    x = layers.Dense(64, activation='relu')(x)
    y_out = layers.Dense(3, activation='softmax')(x)
    self.model = models.Model(x_in, y_out)
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  def trainData(self):
    self.y_train = self.dataTrain['label']
    self.dic_y_mapping = {n:label for n,label in enumerate(np.unique(self.y_train))}
    inverse_dic = {v:k for k,v in self.dic_y_mapping.items()}
    self.y_train = np.array([inverse_dic[y] for y in self.y_train])
    
    self.model.fit(x=self.x_train, y=self.y_train, batch_size=256, epochs=10, shuffle=True, verbose=0, validation_split=0.3)
    self.y_test = self.dataTest['label']
    self.predicted_prob = self.model.predict(self.x_test)
    self.predicted = [self.dic_y_mapping[np.argmax(pred)] for pred in self.predicted_prob]

  def getMetric(self):
    classes = np.unique(self.y_test)
    y_test_array = pd.get_dummies(self.y_test, drop_first=False).values
    
    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(self.y_test, self.predicted)
    print("Accuracy:",  round(accuracy,2))
    print("Detail:")
    print(metrics.classification_report(self.y_test, self.predicted))