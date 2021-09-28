from sklearn import feature_extraction, feature_selection, naive_bayes, pipeline, metrics
import pandas as pd

class Shallow:
    def __init__(self, dataTrain, dataTest):
        self.dataTrain = dataTrain
        self.dataTest = dataTest

        self.addCleanText()
        self.extractFeature()
        self.selectFeature()
        self.updateVectorizer()
        self.trainData()

    def addCleanText(self):
        text_clean = []
        for list_tokens in self.dataTrain['StemmedToken']:
            temp = ''
            temp = ' '.join(list_tokens)
            text_clean.append(temp)
        self.dataTrain["text_clean"] = text_clean

        text_clean = []
        for list_tokens in self.dataTest['StemmedToken']:
            temp = ''
            temp = ' '.join(list_tokens)
            text_clean.append(temp)
        self.dataTest["text_clean"] = text_clean

    def extractFeature(self):
        self.vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1,2))
        self.corpus = self.dataTrain["text_clean"]
        self.vectorizer.fit(self.corpus)
        self.X_train = self.vectorizer.transform(self.corpus)
        self.dic_vocabulary = self.vectorizer.vocabulary_

    def selectFeature(self):
        y = self.dataTrain["label"]
        self.X_names = self.vectorizer.get_feature_names()
        p_value_limit = 0.95
        self.df_features = pd.DataFrame()

        for cat in np.unique(y):
            chi2, p = feature_selection.chi2(X_train, y==cat)
            df_features = df_features.append(pd.DataFrame(
                        {"feature":X_names, "score":1-p, "y":cat}))
            df_features = df_features.sort_values(["y","score"], 
                            ascending=[True,False])
            df_features = df_features[df_features["score"]>p_value_limit]
    
    def updateVectorizer(self):
        self.vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)

        self.vectorizer.fit(self.corpus)
        self.X_train = self.vectorizer.transform(self.corpus)
        self.y_train = self.dataTrain['label']
        self.dic_vocabulary = self.vectorizer.vocabulary_

    def trainData(self):
        classifier = naive_bayes.MultinomialNB()

        ## pipeline
        self.model = pipeline.Pipeline([("vectorizer", self.vectorizer),  
                                ("classifier", classifier)])

        ## train classifier
        self.model["classifier"].fit(self.X_train, self.y_train)

        ## test
        self.X_test = self.dataTest["text_clean"].values
        self.y_test = self.dataTest['label']
        self.predicted = model.predict(X_test)
        self.predicted_prob = model.predict_proba(X_test)

    def getMetrics(self):
        classes = np.unique(self.y_test)
        y_test_array = pd.get_dummies(self.y_test, drop_first=False).values

        ## Accuracy, Precision, Recall
        accuracy = metrics.accuracy_score(self.y_test, self.predicted)
        print("Accuracy:",  round(accuracy,2))
        print("Detail:")
        print(metrics.classification_report(self.y_test, self.predicted))