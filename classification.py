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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mpl_toolkits.mplot3d import Axes3D


vectorizer = TfidfVectorizer(analyzer = "word",ngram_range=[1,1],stop_words = 'english',max_features = 5000) 
#classification
pre_shuffled_data = []
reviews = []
sentiments=[]
for i in xrange(260):
    with open('data/data_'+str(i*20)+'.txt') as data_file:    
        data = json.load(data_file)
    for review in data:
        pre_shuffled_data.append(review)
    print 'first '+ str(i*20) + 'reviews added to array'
#shuffle data
shuffle(pre_shuffled_data)
for review in pre_shuffled_data:
    reviews.append(review['content'])
    sentiments.append(int(review['rating'][0]))

data_features = vectorizer.fit_transform(reviews)
data_features = data_features.toarray()
sentiments = np.asarray(sentiments)

#seperate training data and testing data
TRAINING_NUM =4000
#shuffle features

training_data_features = data_features[:TRAINING_NUM]
testing_data_features = data_features[TRAINING_NUM:]
training_sentiments =  sentiments[:TRAINING_NUM]
testing_sentiments = sentiments[TRAINING_NUM:]
'''
clf = MultinomialNB().fit(training_data_features, training_sentiments)
predicted = clf.predict(testing_data_features)
'''

'''
pct = Perceptron().fit(training_data_features, training_sentiments)
predicted = pct.predict(testing_data_features)

'''
'''
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(training_data_features, training_sentiments)
predicted = clf.predict(testing_data_features)'''

'''clf = DecisionTreeClassifier(random_state=0)
clf.fit(training_data_features, training_sentiments)
predicted = clf.predict(testing_data_features)'''

'''clf = RandomForestClassifier()
clf.fit(training_data_features, training_sentiments)
predicted = clf.predict(testing_data_features)
'''

cnt=0
true =0
worse = 0 # reviews being predicted worse than its actual score
better = 0# reviews being predicted better than its actual score
confusionMatrix= [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
for i,(target, prediction) in enumerate(zip(testing_sentiments, predicted)):
    confusionMatrix[5-target][5-prediction]+=1
    if target == prediction:
        true+=1
    elif target > prediction:
        worse+=1
    else:
        better+=1
    cnt+=1
print true,cnt,1.0*true/cnt
print confusionMatrix

