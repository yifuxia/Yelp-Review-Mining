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

vectorizer = TfidfVectorizer(analyzer = "word",ngram_range=[1,1],stop_words = 'english',max_features = 5000) 


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

#Topic Discovery
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
nmf = NMF(n_components=5, random_state=1).fit(data_features)
print("\nTopics in NMF model:")
feature_names = vectorizer.get_feature_names()
print_top_words(nmf, feature_names, 10)

