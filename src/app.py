import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import tweepy

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator

import nltk 
import string
import re
from PIL import Image
import base64
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import os

nltk.download('stopwords')

consumer_key = os.getenv('CONSUMER_KEY')
consumer_secret = os.getenv('CONSUMER_SECRET')
access_token = os.getenv('ACCESS_TOKEN')
access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit = True)

def tweet_crawler(num_of_tweets=10, real_id="ChicagoFire"):
    '''
        Crawls the tweets using tweets API
    '''
    tweets= []
    likes = []
    time = []
    tweet_id = []
    retweet_count = []
    entities = []
    
    for tweet in tweepy.Cursor(api.user_timeline, id=real_id, tweet_mode="extended").items(num_of_tweets):
        tweets.append(tweet.full_text)
        likes.append(tweet.favorite_count)
        time.append(tweet.created_at)
        tweet_id.append(tweet.id)
        retweet_count.append(tweet.retweet_count)
        entities.append(tweet.entities)


    tweets_df = pd.DataFrame({'id': tweet_id, 'tweets': tweets, 'likes': likes, 'created_at': time, 'num_retweets': retweet_count})
    entities_df = pd.DataFrame(entities)
    
    frames = [tweets_df, entities_df]
    
    final_df = pd.concat(frames, axis=1)
    
    return final_df


#helper functions
def clean_text(text):
    '''
        Removing all the links
    '''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def remove_emoji(text):
    '''
        Removing emojis
    '''
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    '''
        Removes all the stop words
    '''
    text = text.split()
    text = [word for word in text if word not in stopword]
    return text

def cleaning_data(tweet_data_text):
    '''
        Driver that cleans the code
    '''
    
    #Clean the data
    tweet_data_text['tweets'] = tweet_data_text['tweets'].apply(clean_text)
    
    #Remove emojis
    tweet_data_text['tweets'] = tweet_data_text['tweets'].apply(remove_emoji)
    
    #Remove stop words
    tweet_data_text['tweets'] = tweet_data_text['tweets'].apply(remove_stopwords)
    
    return tweet_data_text['tweets']

def vis_clean_data(cleaned_data):
    '''
        Visulizes the data
    '''
    all_words = []
    for text in cleaned_data:
        for word in text:
            all_words.append(word)
    plt.figure(figsize=(30,30))
    all_words_string = " ".join(text for text in all_words)
    wordcloud = WordCloud(background_color="black").generate(all_words_string)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.savefig('WC.jpg')
    img= Image.open("WC.jpg")
    return img

def get_table_download_link(df):
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file make sure you add the .csv extension</a>'
    return href

def key_search(num_of_tweets=10, query="cffc"):
    '''
        Key serach
    '''
    tweets= []
    
    for tweet in tweepy.Cursor(api.search, q=query, tweet_mode="extended").items(num_of_tweets):
        tweets.append(tweet.full_text)

    tweets_df = pd.DataFrame(tweets, columns=['tweets'])
    
    return tweets_df

def freq(cleaned_data):
    '''
        Frequency
    '''
    all_words = []
    for text in cleaned_data:
        for word in text:
            all_words.append(word)
    all_words_string = " ".join(text for text in all_words)
    fdist = FreqDist(word for word in word_tokenize(all_words_string))
    freq_df = pd.DataFrame.from_dict(fdist, orient='index')
    freq_df.columns = ['Frequency']
    freq_df.index.name = "word"
    freq_df = freq_df.sort_values(by="Frequency", ascending=False)

    return freq_df



def app():
    '''
        Streamlit app
    '''
    st.title('Twitter analysis tool')

    activities=["Data visualization","key word search"]

    choice = st.sidebar.selectbox("Select Your Activity",activities)

    if choice == "Data visualization":

        raw_text = st.text_area("Enter the exact twitter handle of the Personality (without @)")

        count = st.number_input("Number of tweets")

        Analyzer_choice = st.selectbox("Select the Activities",  ["Show Tweets","Generate WordCloud", "frequency table" ])


        tweets_df = tweet_crawler(real_id=raw_text, num_of_tweets=int(count))

        if st.button("Fetch"):
            if Analyzer_choice == "Show Tweets":
                st.write(tweets_df.iloc[:, :3])
                st.markdown(get_table_download_link(tweets_df), unsafe_allow_html=True)
            
            elif Analyzer_choice == "Generate WordCloud":
                tweet_data_text = tweets_df.copy()
                cleaned_data = cleaning_data(tweet_data_text)
                img = vis_clean_data(cleaned_data)
                st.image(img)

            elif Analyzer_choice == "frequency table":
                tweet_data_text = tweets_df.copy()
                cleaned_data = cleaning_data(tweet_data_text)
                freq_table = freq(cleaned_data)
                st.write(freq_table)
                st.markdown(get_table_download_link(freq_table), unsafe_allow_html=True)

    elif choice == "key word search":
        raw_text = st.text_area("Enter the exact twitter search (without #)")

        count = st.number_input("Number of tweets")

        Analyzer_choice = st.selectbox("Select the Activities",  ["Show Tweets","Generate WordCloud", "frequency table" ])

        search_df = key_search(query=raw_text, num_of_tweets=int(count))

        if st.button("Fetch"):
            if Analyzer_choice == "Show Tweets":
                st.write(search_df)
                st.markdown(get_table_download_link(search_df), unsafe_allow_html=True)
            
            elif Analyzer_choice == "Generate WordCloud":
                tweet_data_text = search_df.copy()
                cleaned_data = cleaning_data(tweet_data_text)
                img = vis_clean_data(cleaned_data)
                st.image(img)

            elif Analyzer_choice == "frequency table":
                tweet_data_text = search_df.copy()
                cleaned_data = cleaning_data(tweet_data_text)
                freq_table = freq(cleaned_data)
                st.write(freq_table)
                st.markdown(get_table_download_link(freq_table), unsafe_allow_html=True)

if __name__ == "__main__":
    app()
