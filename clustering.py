
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
#for clustering
vectorizer = TfidfVectorizer(analyzer = "word",ngram_range=[1,1],stop_words = 'english',max_features = 5000) 

#clustering
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

#dimention reduction
svd = TruncatedSVD(10)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(data_features)

#apply kmeans
km = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1)
km.fit(data_features)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.cluster_centers_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
feature_names = vectorizer.get_feature_names()
print_top_words(km,feature_names, 10)


'''
_________________________
THIS IS THE TOPIC DISCOVERY RESULT FROM KMEANS:

Topic #0:
love place portos balls potato cakes pastries porto sandwiches lines
Topic #1:
cheese rolls guava potato balls good pastries porto roll place
Topic #2:
line place food good porto long time just wait order
Topic #3:
cake best potato sandwich balls good amazing cuban place porto
Topic #4:
great food place service prices pastries amazing long good sandwiches

'''

'''
# plot reduced dimention features
x=[]
y=[]
z=[]
for idx,i in enumerate(X):
    if i[0]==0:
        print inspect[idx]
    x.append(i[0])
    y.append(i[1])
    z.append(int(inspect[idx]['rating'][0]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r')
plt.show()'''

