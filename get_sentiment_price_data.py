"""
def dict_format(post):
  #Helper function for converting PRAW objects to python dictionary
    result = {}
    for k, v in post.__dict__.items():
        if k.startswith('_'):
            continue
        if k in {'author', 'subreddit'}:
            result[k] = str(v)
            continue
        if v is None:
            continue
        result[k] = v
    return result

submissions = reddit.subreddit('Vechain').top(limit=500)

# load into pandas
subs = pd.DataFrame(dict_format(post) for post in submissions)

# change the `created_utc` column to a datetime object
subs['created_utc'] = pd.to_datetime(subs.created_utc, unit='s')



subs['sentiment'] = subs['title'].apply(sent_calc)

for time_period, group in subs.groupby(pd.Grouper(key='created_utc', freq='d')):
    titles = group.title.values
    # do something with titles...
    print(time_period, len(group)"""
    
     
          
          
# Pull the crypto currency prices using cmc API

from cmc import coinmarketcap
from datetime import datetime



# Select cryptocurrency 

crypto = 'VeChain'
start_date, end_date = datetime(2016,1,1), datetime(2018,12,1)          

vechain_df = coinmarketcap.getDataFor(crypto, start_date, end_date)


#



### Crate empty dictionary for storing all Reddit data


dict = { "title":[],
                "subreddit":[],
                "score":[], 
                "id":[], 
                "url":[], 
                "comms_num": [], 
                "created": [], 
                "body":[]}


# Pull last 100000 posts regarding cryptocurrency of choice

for submission in reddit.subreddit(crypto).top(limit=10000):
    dict["title"].append(submission.title)
    dict['subreddit'].append(submission.subreddit)
    dict["score"].append(submission.score)
    dict["id"].append(submission.id)
    dict["url"].append(submission.url)
    dict["comms_num"].append(submission.num_comments)
    dict["created"].append(submission.created)
    dict["body"].append(submission.selftext)

reddit_df = pd.DataFrame(dict)

reddit_df['created_utc'] = pd.to_datetime(reddit_df.created, unit='s')

reddit_df[['polarity', 'subjectivity']] = reddit_df['title'].apply(lambda Text: pd.Series(textblob.TextBlob(Text).sentiment))

reddit_df_ex = reddit_df[['created_utc', 'polarity']]

reddit_df_ex = reddit_df_ex.set_index('created_utc')

grouped_df = reddit_df_ex.resample('D').mean()

grouped_df = grouped_df.reset_index()

vechain_df = vechain_df.reset_index()

import pandas as pd

merged_df = pd.merge(vechain_df['Open'], grouped_df, left_on = 'Date', right_on = 'created_utc', how = 'inner')

merged_df = merged_df[['created_utc', 'vechain_Open', 'polarity' ]]


merged_df.to_csv('sent_price_file.csv', sep = ',')



