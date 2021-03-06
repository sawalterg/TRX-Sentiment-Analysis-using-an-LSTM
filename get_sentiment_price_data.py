# Pull the crypto currency prices

import pandas as pd
import requests
from bs4 import BeautifulSoup


# Select cryptocurrency 
crypto = 'tron'
start_date, end_date = '20170101', '20181231'       

url = 'https://coinmarketcap.com/currencies/{0}/historical-data/?start={1}&end={2}'.format(crypto, start_date, end_date)
  
content = requests.get(url).content
soup = BeautifulSoup(content, 'html.parser')
table = soup.find('table', {'class': 'table'})


table_form = [[td.text.strip() for td in tr.findChildren('td')] 
        for tr in table.findChildren('tr')]

df_crypt = pd.DataFrame(table_form)
df_crypt.drop(df_crypt.index[0], inplace=True) # remove the missing first row
df_crypt[0] =  pd.to_datetime(df_crypt[0]) # convert to datetime format
for i in range(1,7):
    df_crypt[i] = pd.to_numeric(df_crypt[i].str.replace(",","").str.replace("-","")) # replace missing volume
df_crypt.columns = ['Date','Open','High','Low','Close','Volume','Market Cap']
df_crypt.set_index('Date',inplace=True)
df_crypt.sort_index(inplace=True)



### Create empty dictionary for storing all Reddit data

import praw

from reddit_creds import *

reddit = praw.Reddit(client_id = 'oCrzM31o8070bA',
                     client_secret='kBrPF2ILOwtZVwLlUv-5ELHwRpE', 
                     username = user_name_red,
                     password= password_red,
                     user_agent= 'anything')
dict = { "title":[],
                "subreddit":[],
                "score":[], 
                "id":[], 
                "url":[], 
                "comms_num": [], 
                "created": [], 
                "body":[]}


# Pull last 100000 posts regarding cryptocurrency of choice

for submission in reddit.subreddit('Tronix').top(limit=10000):
    dict["title"].append(submission.title)
    dict['subreddit'].append(submission.subreddit)
    dict["score"].append(submission.score)
    dict["id"].append(submission.id)
    dict["url"].append(submission.url)
    dict["comms_num"].append(submission.num_comments)
    dict["created"].append(submission.created)
    dict["body"].append(submission.selftext)

reddit_df = pd.DataFrame(dict)

from textblob import TextBlob

# Change to datetime format

reddit_df['created_utc'] = pd.to_datetime(reddit_df.created, unit='s')

# Use textblob's pretrained polarity method to determine the sentiment of title

reddit_df[['polarity', 'subjectivity']] = reddit_df['title'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))

# Retrieve relevent columns

reddit_df_ex = reddit_df[['created_utc', 'polarity', 'subjectivity']]

# Switch to index for resampling to a day

reddit_df_ex = reddit_df_ex.set_index('created_utc')

# Take the average sentiment of the day using the mean method

grouped_df = reddit_df_ex.resample('D').mean()

# Reset index

grouped_df = grouped_df.reset_index()


#grouped_df['created_utc'] = pd.to_datetime(grouped_df.created_utc, unit='s')


# Create new index

df_crypt = df_crypt.reset_index()



merged_df = pd.merge(df_crypt, grouped_df, left_on = 'Date', right_on = 'created_utc', how = 'inner')

merged_df = merged_df[['Date', 'Close', 'Volume', 'polarity', 'subjectivity' ]]

merged_df['target'] = merged_df.Close.shift(-1)


# Make final csv

merged_df.to_csv('sent_price_file.csv', sep = ',')



