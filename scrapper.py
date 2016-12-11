from bs4 import BeautifulSoup
import urllib
import json

NUM_PAGES = 260

for i in xrange(NUM_PAGES):
    data=[]
    url = 'https://www.yelp.com/biz/portos-bakery-and-cafe-burbank?start=' + str(i*20) 
    r = urllib.urlopen(url).read()
    soup = BeautifulSoup(r,"html.parser")
    reviews = soup.find_all("div", class_="review review--with-sidebar")
    for review in reviews:
        obj ={
            'name':None,
            'time':None,
            'content':None,
            'rating':None
        }
        obj['name'] = review.find('a', class_='user-display-name').getText().encode('utf-8')  
        obj['time'] = review.find("span",class_='rating-qualifier').getText().encode('utf-8').strip()
        obj['content'] = review.p.getText().encode('utf-8')
        obj['rating'] = review.find('div', class_='i-stars')["title"].encode('utf-8')
        data.append(obj)
    with open('data/data_'+str(i*20)+'.txt', 'w') as outfile:
        json.dump(data, outfile)
    print 'data_'+str(i*20)+'.txt'+' successfully dumped!'