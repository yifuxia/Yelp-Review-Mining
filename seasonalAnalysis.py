import json
from datetime import date,datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             max_features = 5000) 
'''

# Get seasonal data

seasons ={
	'spring':[],
	'summer':[],
	'fall':[],
	'winter':[]
}



for i in xrange(260):
	with open('data/data_'+str(i*20)+'.txt') as data_file:    
		data = json.load(data_file)
	for review in data:
		try:
			doy = datetime.strptime(review['time'], '%m/%d/%Y').timetuple().tm_yday
			# "day of year" ranges for the northern hemisphere
			spring = range(80, 172)
			summer = range(172, 264)
			fall = range(264, 355)
			# winter = everything else

			if doy in spring:
			  season = 'spring'
			elif doy in summer:
			  season = 'summer'
			elif doy in fall:
			  season = 'fall'
			else:
			  season = 'winter'
			print season
			seasons[season].append((review['content'],int(review['rating'][0]),review['time']))
		except:
			pass
for key in seasons:
	reviews = []
	ratings = []
	#get all reviews and ratings in this season
	for review in seasons[key]:
		reviews.append(review[0])
		ratings.append(review[1])
	print key,sum(ratings)*1.0/len(ratings)
'''
# Get monthly data

months = [-1,[],[],[],[],[],[],[],[],[],[],[],[]]
for i in xrange(260):
	with open('data/data_'+str(i*20)+'.txt') as data_file:    
		data = json.load(data_file)
	for review in data:
		try:
			months[datetime.strptime(review['time'], '%m/%d/%Y').date().month].append(int(review['rating'][0]))
		except:
			pass
with open('result/monthlyRating.csv', 'w') as outfile:
	outfile.write('month,rating\n')
	for i in xrange(1,len(months)):
		outfile.write(str(i)+','+str(sum(months[i]) * 1.0 / len(months[i]))+'\n') 

'''

	# Top 10 words in each season


	data_features = vectorizer.fit_transform(reviews)
	data_features = data_features.toarray()
	# Take a look at the words in the vocabulary
	vocab = vectorizer.get_feature_names()

	# Sum up the tdidf value of each vocabulary word
	dist = np.sum(data_features, axis=0)
	wordCount = []
	for tag, count in zip(vocab, dist):
	    wordCount.append((tag, count))
	wordCount.sort(key = lambda (a,b):-b)
	wordCount = wordCount[:100]
	with open('result/'+key+'-wordFrequency.csv', 'w') as outfile:
		outfile.write('word,tf-idf\n')
		for t in wordCount:
			try:
				outfile.write(str(t[0])+','+str(t[1])+'\n')
			except:
				pass
'''