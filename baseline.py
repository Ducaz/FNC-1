# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:03:39 2017

@author: YIXUAN LI
"""
import numpy as np
from nltk import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import hstack

from utils.dataset import DataSet
from utils.generate_test_splits import split
from utils.score import report_score

dataset = DataSet()
data_splits = split(dataset)

training_data = data_splits['training']
dev_data = data_splits['dev']
test_data = data_splits['test']

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

class Preprocessor(object):
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopwords_eng = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()
        
    def __call__(self, doc):
        return [self.lemmatizer.lemmatize(t) for t in self.tokenizer.tokenize(doc)]
        
    def process(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        tokens_processed = []
        for t in tokens:
            if t in self.stopwords_eng: continue
            tokens_processed.append(self.lemmatizer.lemmatize(t))
        return tokens_processed
        
class Document(object):
    def __init__(self, data):
        self.stances = []
        self.headlines = []
        self.body_texts = []
        self.size = 0
        for dict_item in data:
            label_index = LABELS.index(dict_item['Stance'])
            headline = dict_item['Headline']
            body = dataset.articles[dict_item['Body ID']]
            self.stances.append(label_index)
            self.headlines.append(headline)
            self.body_texts.append(body)
        self.size = len(self.stances)
        self.stances = np.asarray(self.stances)
        
    def get_full_text(self):
        full_texts = []
        for i in range(self.size):
            text = '\n'.join((self.headlines[i], self.body_texts[i]))
            full_texts.append(text)
        return full_texts


if __name__ == '__main__':
    #preprocessor = Preprocessor()
    training_doc = Document(training_data)
    test_doc = Document(test_data)
    
    vectorizer = CountVectorizer(ngram_range=(1,2), min_df=2, 
                                 stop_words='english')
    train_headline = vectorizer.fit_transform(training_doc.headlines)
    test_headline = vectorizer.transform(test_doc.headlines)
    train_body = vectorizer.fit_transform(training_doc.body_texts)
    test_body = vectorizer.transform(test_doc.body_texts)
    
    ch2 = SelectKBest(chi2, k=1000)
    ch2.fit(train_headline, training_doc.stances)
    train_headline = ch2.transform(train_headline)
    test_headline = ch2.transform(test_headline)
    ch2.fit(train_body, training_doc.stances)
    train_body = ch2.transform(train_body)
    test_body = ch2.transform(test_body)
    
    train_features = hstack((train_headline, train_body))
    test_features = hstack((test_headline, test_body))
    
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(train_features, training_doc.stances)
    
    prediction = classifier.predict(test_features)
    
    actual_label = [LABELS[x] for x in test_doc.stances]
    predicted_label = [LABELS[x] for x in prediction]
    report_score(actual_label, predicted_label)
