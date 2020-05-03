import pandas as pd

file = pd.read_csv('../Tweets.csv')
data = pd.DataFrame(file, columns=['tweet_id', 'airline_sentiment', 'text'])
data.set_index('tweet_id', drop=False)

decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]