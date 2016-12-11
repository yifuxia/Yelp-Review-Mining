import re
import json
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

#for word count
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             max_features = 5000) 
reviews = []
sentiments = []
for i in xrange(260):
    with open('data/data_'+str(i*20)+'.txt') as data_file:    
        data = json.load(data_file)
    for review in data:
        reviews.append(review['content'])
        sentiments.append(int(review['rating'][0]))
    print 'first '+ str(i*20) + 'reviews added to array'

data_features = vectorizer.fit_transform(reviews)
data_features = data_features.toarray()

weighted_features = []
for i in xrange(len(data_features)):
    weighted_features.append(data_features[i]*sentiments[i])
word_rating_dict = np.sum(weighted_features,axis=0)*1.0 / np.sum(data_features,axis=0)
vocab = vectorizer.get_feature_names()
wordRating =[]
dist = np.sum(data_features, axis=0)
print dist
for i in xrange(len(vocab)):
    wordRating.append((vocab[i],round(word_rating_dict[i],3),dist[i]))
wordRating.sort(key = lambda (a,b,c):-b)
with open('result/wordRatings_wz_Counts.csv', 'w') as outfile:
    outfile.write('word,rating,count\n')
    for t in wordRating:
            try:
                outfile.write(str(t[0])+','+str(t[1])+','+str(t[2])+'\n')
            except:
                pass
