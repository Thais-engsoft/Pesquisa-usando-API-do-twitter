import tweepy
import time
import re 
import squarify
import collections
from textblob import TextBlob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium import plugins
import warnings
from googletrans import Translator
from PIL import Image
from unidecode import unidecode
from geopy.geocoders import Nominatim
from wordcloud import wordcloud, STOPWORDS
import source

#Chaves de acesso a api
key_consumer = ''
consumer_secret = ''
access = ''
access_secret = ''

#Estabelecendo conexão
class TweetAnalyzer():
 
  def __init__ (self, key_consumer, consumer_secret, access, access_secret):
        auth = tweepy.OAuthHandler(key_consumer, consumer_secret)
        auth.set_access_token(access, access_secret)
 
 self.conToken = tweepy.API(auth, wait_on_rate_limit_notify=True, wait_on_rate_limit=True, retry_count=4, retry_delay=7)
 
#Limpeza dos tweets

def __clean_tweet(self, tweets_text):
        clean_text = re.sub(r'RT+', '', tweets_text)
        clean_text = re.sub(r'@\S+', '', clean_text)
        clean_text = re.sub(r'http\S+', '', clean_text)
        clean_text = clean_text.replace("\n", " ")
        return clean_text
 def search_by_keyword(self, keyword, count=7, result_type='mixed', lang='en', tweet_mode='extended'):
        tweets_iter = tweepy.Cursor(self.conToken.search,
        q=keyword, tweet_mode=tweet_mode, rpp=count, 
        result_type=result_type, since=datetime(2021,5,21,0,0,0).date(),
        lang=lang, include_entities=True).items(count)
 
        return tweets_iter
 def prepare_tweets_list(self, tweets_iter):
    tweets_data_list = []
 
    for tweet in tweets_iter:
 
        if not 'retweeted_status' in dir(tweet):
           tweet_text = self.__clean_tweet(tweet.full_text)
           tweets_data = {
           'len' : len(tweet_text),
           'ID' : tweet.id,
           'UserName' : tweet.user.screen_name,
           'UserLocation' : tweet.user.name,
           'TweetText' : tweet_text,
           'Language' : tweet.user.lang,
           'Date' : tweet.created_at,
           'Sources' : tweet.source,
           'Likes' : tweet.favorite_count,
           'Retweets' : tweet.retweet_count,
           'Coordinates' : tweet.coordinates,
           'Place' : tweet.place 
            }
           tweets_data_list.append(tweets_data)
 
    return tweets_data_list
 
  
#Análise de sentimentos é feita por meio de mineração textual
  
  def sentiment_polarity(self, tweets_text_list):
    tweets_sentiments_list = []
    for tweet in tweets_text_list:
      polarity = TextBlob(tweet).sentiment.polarity
 
      if polarity > 0:
        tweets_sentiments_list.append('Positive')
      elif polarity < 0:
        tweets_sentiments_list.append('Negative')
      else:
        tweets_sentiments_list.append('Neutral')
 
    return tweets_sentiments_list 

analyzer = TweetAnalyzer(key_consumer = key_consumer, consumer_secret = consumer_secret, access = access, access_secret = access_secret)
keyword = (" '#BTSARMY' or 'BTS' ")
count = 5000

tweets_iter = analyzer.search_by_keyword(keyword, count)
tweets_list = analyzer.prepare_tweets_list(tweets_iter)
tweets_df = pd.DataFrame(tweets_list)
print(tweets_df)

like_max = np.max(tweets_df['Likes'])
likes = tweets_df[tweets_df.Likes == like_max].index[0]
print(f"O tweet com mais curtidas é: {tweets_df['TweetText'][likes]}")
print(f"Quantidade de curtidas: {like_max}")
 
retweet_max = np.max(tweets_df['Retweets'])
retweet = tweets_df[tweets_df.Retweets == retweet_max].index[0]
print(f"O tweet com mais retweets é: {tweets_df['TweetText'][retweet]}") 
print(f"Quantidade de retweets: {retweet_max}")


#Formação da nuvem de palavras que mais se fizeram presentes nos tweets 

from wordcloud import WordCloud
words = ' '.join(tweets_df['TweetText'])
 
words_clean = " ".join([word for word in words.split()])
 
warnings.simplefilter('ignore')
 
mask = np.array(Image.open('crown.png'))
wc = WordCloud(stopwords=STOPWORDS, mask=mask,
               max_words=3000, max_font_size=100,
               min_font_size=10, random_state=42,
               background_color='white', mode="RGB",
               width=mask.shape[1], height=mask.shape[0],
               normalize_plurals=True).generate(words_clean)
 
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.savefig('Buter', dpi=300)
plt.show()


#Gráfico onde mostra em quais dispositivos tiveram o maior número de tweets

source_list = tweets_df['Sources'].tolist()
occurrences = collections.Counter(source_list)
source_df = pd.DataFrame({'Total':list(occurrences.values())}, index=occurrences.keys())
sources_sorted = source_df.sort_values('Total', ascending=True)
 
plt.style.use('ggplot')
plt.rcParams['axes.edgecolor']='#230403'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#280801'
plt.rcParams['ytick.color']='#430663'
my_range=list(range(1,len(sources_sorted.index)+1))
ax = sources_sorted.Total.plot(kind='barh',color='#d75cf2', alpha=0.8, linewidth=5, figsize=(15,15))
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.savefig('source_tweets.png', bbox_inches='tight', pad_inches=0.5)

tweets_df['Sentiment'] = analyzer.sentiment_polarity(tweets_df['TweetText'])
sentiment_percentage = tweets_df.groupby('Sentiment')['ID'].count().apply(lambda x : 100 * x / count)
sentiment_percentage.plot(kind='bar')
plt.show()
plt.savefig('sentiments_tweets.png', bbox_inches='tight', pad_inches=0.5)


#Mapa de calor, com a localização dos maiores pontos de tweests no momento solicitado
geolocator = Nominatim(user_agent="TweeterSentiments")
latitude = []
longitude = []
 
for user_location in tweets_df['UserLocation']:
  try:
    location = geolocator.geocode(user_location)
    latitude.append(location.latitude)
    longitude.append(location.longitude)
  except:
    continue
 
coordenadas = np.column_stack((latitude, longitude))
 
mapa = folium.Map(zoom_start=3.)
mapa.add_child(plugins.HeatMap(coordenadas))
mapa.save('Mapa_calor_tweets1.html')


data = tweets_df
data['Date'] = pd.to_datetime(data['Date']).apply(lambda x: x.date())
tlen = pd.Series(data['Date'].value_counts(), index=data['Date'])
 
tlen.plot(figsize=(16,4), color='b')
plt.savefig('timeline_tweets.png', bbox_inches='tight', pad_inches=0.5)
