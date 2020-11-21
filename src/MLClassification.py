import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import *
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN

DECODE_MAP = {"negative": 0, "neutral": 1, "positive": 2}
STOP_WORDS = nltk.corpus.stopwords.words('english')


def decode_sentiment(label):
    return DECODE_MAP[str(label)]


def cleanseTweets(tweet):
    withoutAuthors = re.sub("@[\\w]+\\s", "", tweet)
    withoutDigitsAndOthers = re.sub("[^a-zA-Z ']", "", withoutAuthors)
    withoutLinks = re.sub("http\\S+", "", withoutDigitsAndOthers)
    withoutWhiteSpacesStartEnd = withoutLinks.strip()
    cleanedTweet = re.sub("\\s+", " ", withoutWhiteSpacesStartEnd)
    return cleanedTweet


def preprocessTweets(tweet):
    global stemmer
    print(tweet)
    # tokenization of tweets
    tweet = tweet.split()
    print(tweet)

    # Transform tokens to lower case
    tweet = [token.lower() for token in tweet]
    print(tweet)

    # Removing tokens of lenght <= 4 and <= 20
    tweet = list(filter(lambda token: 4 <= len(token) <= 20, tweet))
   # tweet = tweet.apply(list(filter(lambda token: 4 <= len(token) <= 20, tweet)))
    print(tweet)

    # Removing stop words
    tweet = list(filter(lambda token: token not in STOP_WORDS, tweet))
    #tweet = tweet.apply(list(filter(lambda token: token not in STOP_WORDS, tweet)))
    print(tweet)

    # Applying stemming
    stemmer = PorterStemmer()
    tweet = [stemmer.stem(token) for token in tweet]
    #tweet = tweet.apply(lambda token: stemmer.stem(token))
    print(tweet)

    # Joining tokens into stream
    seperator = ' '
    tweet = ' '.join(tweet)
    #tweet = tweet.apply(lambda x: ' '.join(x))
    print(tweet)

    return tweet



# read file
file = pd.read_csv('../Tweets.csv')
data = pd.DataFrame(file, columns=['tweet_id', 'airline_sentiment', 'text'])
data.set_index('tweet_id', drop=False)

print(data.head(10))

# convert target from string labels to int
data.airline_sentiment = data.airline_sentiment.apply(lambda x: decode_sentiment(x))

# tweets cleansing (remove specials characters, links, authors, digits and whitespaces)
data['cleansedTweets'] = data['text'].apply(lambda x: cleanseTweets(x))
print(data['cleansedTweets'].head())
print(data.to_csv('cleanseTweets.csv'))

sentiment = [2]
positiveTweets = data[data.airline_sentiment.isin(sentiment)]
sentiment = [1]
neutralTweets = data[data.airline_sentiment.isin(sentiment)]
sentiment = [0]
negativeTweets = data[data.airline_sentiment.isin(sentiment)]

positiveTweetsSample = positiveTweets.sample(2000)
neutralTweetsSample = neutralTweets.sample(2000)
negativeTweetsSample = negativeTweets.sample(2000)

#data = pd.concat([positiveTweetsSample, neutralTweetsSample, negativeTweetsSample])

print(positiveTweets['airline_sentiment'].head())
#data = data.sample(5000)
data['cleansedTweets'] = data['cleansedTweets'].apply(lambda x: preprocessTweets(x))

print(data.head())

tfidf = TfidfVectorizer()
features = tfidf.fit_transform(data['cleansedTweets'])
print(features.shape)

X_train, X_test, y_train, y_test = train_test_split(features,data['airline_sentiment'], test_size = 0.2, stratify = data['airline_sentiment'], random_state = 42)


# Balancing training dataset with SMOTE method using oversampling
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
print(sorted(Counter(y_train).items()))


from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
#clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
clf = svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear')
clf.fit(X_train, y_train)

prediction_svm = clf.predict(X_test)
print(accuracy_score(prediction_svm, y_test))
print(confusion_matrix(prediction_svm, y_test, labels=[0, 1, 2]))

from sklearn.linear_model import LogisticRegression
#lr = OneVsRestClassifier(LogisticRegression())
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

prediction_lr = lr.predict(X_test)
print(accuracy_score(prediction_lr,y_test))
print(confusion_matrix(prediction_lr, y_test, labels=[0, 1, 2]))

from sklearn.neural_network import MLPClassifier
#mlp = OneVsRestClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=200))
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=200)
mlp.fit(X_train, y_train)

prediction_mlp = mlp.predict(X_test)
print(accuracy_score(prediction_mlp,y_test))
print(confusion_matrix(prediction_mlp, y_test, labels=[0, 1, 2]))

from sklearn.naive_bayes import MultinomialNB
bayes = MultinomialNB()
bayes.fit(X_train, y_train)

prediction_bayes = bayes.predict(X_test)
print(accuracy_score(prediction_bayes,y_test))
print(confusion_matrix(prediction_bayes, y_test, labels=[0, 1, 2]))

from sklearn import tree
tree = tree.DecisionTreeClassifier()
tree.fit(X_train, y_train)

prediction_tree = tree.predict(X_test)
print(accuracy_score(prediction_tree,y_test))
print(confusion_matrix(prediction_tree, y_test, labels=[0, 1, 2]))

tweets = [
    "What a great airline, the trip was a pleasure!",
    "My issue was quickly resolved after calling customer support. Thanks!",
    "What the hell! My flight was cancelled again. This sucks!",
    "Service was awful. I'll never fly with you again.",
    "You fuckers lost my luggage. Never again!",
    "I have mixed feelings about airlines. I don't know what I think.",
    ""
]

cleanTweets = [cleanseTweets(tweet) for tweet in tweets]
cleanTweets = [preprocessTweets(tweet) for tweet in cleanTweets]
features = tfidf.transform(cleanTweets)

print(clf.predict_proba(features))
print(clf.predict(features))

print(lr.predict_proba(features))
print(lr.predict(features))

print(mlp.predict_proba(features))
print(mlp.predict(features))

print(bayes.predict_proba(features))
print(bayes.predict(features))

print(tree.predict_proba(features))
print(tree.predict(features))

