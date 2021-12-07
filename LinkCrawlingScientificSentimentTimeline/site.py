from flask import Flask
from flask import request
from flask import render_template
import pandas as pd
import subprocess
import json
import time
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
from textblob import TextBlob
from datetime import datetime
import os
import os.path
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
#Loading Dataframes
pro_sci = pd.read_csv('pro_science.csv',sep=',')
anti_sci = pd.read_csv('anti_science.csv')

#Renaming and Cleaning Dataframes
pro_sci = pro_sci.drop(columns='Unnamed: 0')
pro_sci.rename(columns= {'0':'Tweet'}, inplace=True)
#adds label 0 for pro-science Tweets
pro_sci['label'] = 1

anti_sci = anti_sci.drop(columns='Unnamed: 0')
anti_sci.rename(columns= {'0':'Tweet'}, inplace=True)
#adds label 0 for anti-science Tweets
anti_sci['label'] = 0

corpus = pro_sci.append(anti_sci, ignore_index=True)

#Text processing in corpus dataframe
#Lowercase
corpus['Tweet'] = [entry.lower() for entry in corpus['Tweet']]
#Tokenizing
corpus['Tweet']= [word_tokenize(entry) for entry in corpus['Tweet']]
#Tags for Word Lemmatization
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(corpus['Tweet']):
    final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # Provides tags for nouns, verbs, adjectives
    for word, tag in pos_tag(entry):
        # Check for stopwords
        if word not in stopwords.words('english') and word.isalpha():
            word_final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            final_words.append(word_final)
    # Stores final processed words for each Tweet in column "text_final"
    corpus.loc[index,'text_final'] = str(final_words)
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['text_final'],corpus['label'],test_size=0.3)
Tfidf_vect = TfidfVectorizer(ngram_range=(1,3))
Tfidf_vect.fit(corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
#Support Vector Machine
svm_clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
clf = CalibratedClassifierCV(base_estimator=svm_clf) 
clf.fit(Train_X_Tfidf,Train_Y, np.random.seed(42))
svm_pred = clf.predict(Test_X_Tfidf)
#print("SVM Accuracy Score -> ",accuracy_score(svm_pred, Test_Y))

def sci_sentiment(tweet):
    df = pd.DataFrame([tweet],columns=['Tweet'])
    df['Tweet'] = [entry.lower() for entry in df['Tweet']]
    Tfidf_vect.fit(corpus['text_final'])
    tweet_Tfidf = Tfidf_vect.transform((df['Tweet']))
    svm_tweet_pred = clf.predict(tweet_Tfidf)
    #print('SVM:', svm_tweet_pred)
    pred_probs = clf.predict_proba(tweet_Tfidf)
    return pred_probs[0][1] 
useragent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'
tweets = []
app = Flask(__name__)
@app.route('/')
def my_form():
    return render_template('index.html')
@app.route('/',methods=['POST'])
def my_form_post():
    user = request.form['user']
    modulo = request.form['modulo']
    filename = user + '.json'
    if not os.path.exists(filename):
        op = open(filename,'w')
        subprocess.call(['twarc','timeline',user],stdout=op)
    for line in open(filename,'r'):
        tweets.append(json.loads(line))
    urls = []
    dates = []
    tweetcontent = []
    for tweet in tweets:
        if len(tweet['entities']['urls']) != 0: # if tweet has a link store it and the date
            if not ('twitter' in tweet['entities']['urls'][0]['expanded_url']): # filter out twitter links
                if not ('youtube' in tweet['entities']['urls'][0]['expanded_url']): # filter out youtube links
                    if not ('youtu.be' in tweet['entities']['urls'][0]['expanded_url']): # filter out youtube links
                        if not ('tiktok' in tweet['entities']['urls'][0]['expanded_url']): # filter out tiktok links
                            if not ('vimeo' in tweet['entities']['urls'][0]['expanded_url']): # filter out vimeo links
                                urls.append(tweet['entities']['urls'][0]['expanded_url'])
                                dates.append(tweet['created_at'])
                                tweetcontent.append(tweet['full_text'])
    thinnames = []
    thindates = []
    thincontent = []
    for i in range(len(urls)):
        if i % int(modulo) == 0:
            thinnames.append(urls[i])
            thindates.append(dates[i])
            thincontent.append(tweetcontent[i])
    textarr = []
    badurls = []
    changed = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'}
    e = 0
    l = len(thinnames)
    for url in thinnames:
        time.sleep(0.1)
        percent = (e/l)*100
        formatperc = '{:.2f}'.format(percent)
        print(str(formatperc) + '%')
        try:
            page = requests.get(url, headers=headers)
            if page.status_code == 200:
                soup = BeautifulSoup(page.content, 'html.parser')
                textarr.append(soup.get_text())
                e += 1
            else: 
                badurls.append(url)
                textarr.append('error')
                e += 1
        except:
            badurls.append(url)
            textarr.append('error')
            e += 1
    countnormal = 0
    counterror = 0
    for text in textarr:
        if 'error' in text or text == '':
            counterror += 1
        countnormal += 1
    percent = (counterror/countnormal)*100
    formatperc = '{:.2f}'.format(percent)
    ephem = 'ephemerality: ' + str(formatperc) + '%'
    dateformat = '%a %b %d %X %z %Y'
    fdates = []
    for dt in thindates:
        fdates.append(datetime.strptime(dt,dateformat))
    datesfinal = matplotlib.dates.date2num(fdates)
    values = []
    colors = []
    for content in thincontent:
        sent = TextBlob(content).sentiment.polarity
        if sent < -0.5:
            colors.append((168/255,10/255,39/255,.75))
        elif sent >= -0.5 and sent < 0:
            colors.append((181/255,40/255,141/255,.75))
        elif sent == 0:
            colors.append((140/255,137/255,145/255,.75))
        elif sent > 0 and sent <= 0.5:
            colors.append((104/255,21/255,237/255,.75))
        else:
            colors.append((21/255,129/255,237/255,.75))
    for text in textarr:
        if text == 'error':
            values.append(0)
        else:
            values.append((sci_sentiment(text)-.5)*2)
    for i in range(len(colors)):
        plt.plot_date(datesfinal[i],values[i],marker='o',c=colors[i],ms=10)
    plt.title(filename.split('.')[0])
    plt.gcf().autofmt_xdate()
    plt.ylabel('Scientific Sentiment')
    plt.ylim([-1,1])
    plt.savefig('static/photos/'+ user + '.png')
    plt.close()
    folder = os.path.join('static','photos')
    app.config['upload'] = folder
    fullfilename = os.path.join(app.config['upload'],user+'.png')
    arrowpic = os.path.join(app.config['upload'],'arrow.png')
    hmap = os.path.join(app.config['upload'],'heatmap.png')
    return render_template("index.html",image=fullfilename,ephem=ephem,arrow=arrowpic,heatmap=hmap)
if __name__ == '__main__':
    app.run(debug=True)