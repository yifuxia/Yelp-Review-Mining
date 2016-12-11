import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             max_features = 5000) 
reviews = []
sentiments=[]
for i in xrange(260):
	with open('data/data_'+str(i*20)+'.txt') as data_file:    
	    data = json.load(data_file)
	for review in data:
		reviews.append(review['content'])
		sentiments.append(int(review['rating'][0]))
	print 'first '+ str(i*20) + 'reviews added to array'













'''
# Save wordcount result to file

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()

# Sum up the tdidf value of each vocabulary word
dist = np.sum(train_data_features, axis=0)
wordCount = []
for tag, count in zip(vocab, dist):
    wordCount.append((tag, count))
wordCount.sort(key = lambda (a,b):b)
with open('result/wordFrequency', 'w') as outfile:
        for t in wordCount:
        	try:
        		outfile.write(str(t[0])+','+str(t[1])+'\n')
        	except:
        		pass
'''


