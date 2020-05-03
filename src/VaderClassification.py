import pandas as pd
from sklearn.metrics import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

file = pd.read_csv('../Tweets.csv')
data = pd.DataFrame(file, columns=['tweet_id', 'airline_sentiment', 'text'])
data.set_index('tweet_id', drop=False)

analyser = SentimentIntensityAnalyzer()
vaderSentiments = []
data['VADERSentiment'] = 'NA'

for index, row in data.iterrows():
    score = analyser.polarity_scores(row['text'])
    print(str(score))
    if score['compound'] >= 0.05:
        data.at[index, 'VADERSentiment'] = 'positive'
    elif score['compound'] <= -0.05:
        data.at[index, 'VADERSentiment'] = 'negative'
    else:
        data.at[index, 'VADERSentiment'] = 'neutral'

data.to_csv('processedTweets.csv')
print(data.head(10))

confusionMatrix = confusion_matrix(data['airline_sentiment'], data['VADERSentiment'],
                                   labels=['negative', 'neutral', 'positive'])
print(confusionMatrix)

accuracy = accuracy_score(data['airline_sentiment'], data['VADERSentiment'])
print(accuracy)
