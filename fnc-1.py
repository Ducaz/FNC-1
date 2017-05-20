# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:03:39 2017

@author: YIXUAN LI
"""
from __future__ import print_function
import numpy as np
from gensim.models import KeyedVectors
#from keras.preprocessing import sequence
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
#from keras.datasets import imdb
from nltk import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import StandardScaler
#from sklearn.neural_network import MLPClassifier
from scipy.sparse import hstack, csr_matrix

from utils.dataset import DataSet
from utils.generate_test_splits import split
from utils.score import report_score

dataset = DataSet()
data_splits = split(dataset)

training_data = data_splits['training']
dev_data = data_splits['dev']
test_data = data_splits['test']

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
google_data_path = 'GoogleNews-vectors-negative300.bin'
WORD_VECTORS_NUM = 300


class Preprocessor(object):
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopwords_eng = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()
        
    def __call__(self, doc):
        return [self.lemmatizer.lemmatize(t) for t in 
                self.tokenizer.tokenize(doc)]
        
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
    
    def get_all_words(self):
        all_words = set()
        preprocessor = Preprocessor()
        for i in range(self.size):
            w1 = preprocessor.process(self.body_texts[i])
            w2 = preprocessor.process(self.headlines[i])
            [all_words.add(x) for x in w1]
            [all_words.add(x) for x in w2]
        return list(all_words)


def get_word_vectors(all_words):
    model = KeyedVectors.load_word2vec_format(google_data_path, 
                                                            binary=True)
    word_vectors = {}
    for word in all_words:
        if word in model:
            word_vectors[word] = model[word].reshape(1,300)
    return word_vectors


def get_similarity_feature(document, vectorizer):
    sim = np.zeros((document.size, 1))
    for i in range(document.size):
        headline, body = document.headlines[i], document.body_texts[i]
        tfidf = vectorizer.fit_transform([headline, body])
        sim[i] = cosine_similarity(tfidf[0], tfidf[1])
    return sim


def get_word_vectors_feature(document, word_vectors):
    wv = np.zeros((document.size, 1))
    preprocessor = Preprocessor()
    for i in range(document.size):
        headline, body = document.headlines[i], document.body_texts[i]
        headline_wv = np.zeros((1, WORD_VECTORS_NUM))
        body_wv = np.zeros((1, WORD_VECTORS_NUM))
        count = 0
        for x in preprocessor.process(headline):
            if x in word_vectors:
                headline_wv += word_vectors[x]
                count += 1
        if(count): headline_wv /= count
        headline_wv = csr_matrix(headline_wv)
        count = 0
        for x in preprocessor.process(body):
            if x in word_vectors:
                body_wv += word_vectors[x]
        if(count): body_wv /= count
        body_wv = csr_matrix(body_wv)
        wv[i] = cosine_similarity(headline_wv, body_wv)
    return wv
                

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
    
    ch2_headline = SelectKBest(chi2, k=500)
    ch2_headline.fit(train_headline, training_doc.stances)
    train_headline = ch2_headline.transform(train_headline)
    test_headline = ch2_headline.transform(test_headline)
    ch2_body = SelectKBest(chi2, k=1000)
    ch2_body.fit(train_body, training_doc.stances)
    train_body = ch2_body.transform(train_body)
    test_body = ch2_body.transform(test_body)
    
    # cosine similarity
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    train_sim = get_similarity_feature(training_doc, tfidf_vectorizer)
    test_sim = get_similarity_feature(test_doc, tfidf_vectorizer)
    
    # word vectors
    #train_all_words = training_doc.get_all_words()
    #word_vectors = get_word_vectors(train_all_words)
    #train_wv = get_word_vectors_feature(training_doc, word_vectors)
    #test_wv = get_word_vectors_feature(test_doc, word_vectors)
    
    #train_features = hstack((train_wv, train_sim, train_headline, train_body))
    #test_features = hstack((test_wv, test_sim, test_headline, test_body))
    train_features = hstack((train_sim, train_headline, train_body))
    test_features = hstack((test_sim, test_headline, test_body))
    
    classifier = RandomForestClassifier(n_estimators=10, random_state=10)
    classifier.fit(train_features, training_doc.stances)
    
    prediction = classifier.predict(test_features)
    
    actual_label = [LABELS[x] for x in test_doc.stances]
    predicted_label = [LABELS[x] for x in prediction]
    report_score(actual_label, predicted_label)
    
    x_train = train_features.toarray()
    x_test = test_features.toarray()
    y_train = np.asarray(training_doc.stances)
    y_test = np.asarray(test_doc.stances)
