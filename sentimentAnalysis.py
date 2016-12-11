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

from mpl_toolkits.mplot3d import Axes3D

#for word count
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             max_features = 5000) 

# word ratings (positive & negative words discrimination)
reviews = []
inspect = []
for i in xrange(260):
    with open('data/data_'+str(i*20)+'.txt') as data_file:    
        data = json.load(data_file)
    for review in data:
        reviews.append(review['content'])
        inspect.append(review)
    print 'first '+ str(i*20) + 'reviews added to array'

data_features = vectorizer.fit_transform(reviews)
data_features = data_features.toarray()
weighted_features = []
for i in xrange(len(data_features)):
    weighted_features.append(data_features[i]*sentiments[i])
word_rating_dict = np.sum(weighted_features,axis=0) / np.sum(data_features,axis=0) 
vocab = vectorizer.get_feature_names()
wordRating =[]
dist = np.sum(data_features, axis=0)
print dist
for i in xrange(len(vocab)):
    if dist[i]>1:#filter out low freq words
        wordRating.append((vocab[i],word_rating_dict[i]))
wordRating.sort(key = lambda (a,b):-b)
with open('result/wordRatings.csv', 'w') as outfile:
    outfile.write('word,rating\n')
    for t in wordRating:
            try:
                outfile.write(str(t[0])+','+str(t[1])+'\n')
            except:
                pass
    









'''
# Save wordcount result to file

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()

# Sum up the tdidf value of each vocabulary word
dist = np.sum(data_features, axis=0)
wordCount = []
for tag, count in zip(vocab, dist):
    wordCount.append((tag, count))
wordCount.sort(key = lambda (a,b):-b)
wordCount = wordCount[:100]
with open('result/wordFrequency.csv', 'w') as outfile:
        outfile.write('word,tf-idf\n')
        for t in wordCount:
            try:
                outfile.write(str(t[0])+','+str(t[1])+'\n')
            except:
                pass
'''


