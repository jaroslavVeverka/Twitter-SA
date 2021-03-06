import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from stop_words import get_stop_words

src_dir = os.path.join(os.getcwd(), os.pardir, 'src')
sys.path.append(src_dir)

from src.multilabel import multilabel_train_test_split
from src.SparseInteractions import SparseInteractions
from src.LemmaTokenizer import LemmaTokenizer

data = pd.read_csv('../Tweets.csv')
print(data.head())

le = LabelEncoder()
data['airline_sentiment'] = le.fit_transform(data['airline_sentiment'])

print(data[['airline_sentiment']].head())

X_train, X_test, y_train, y_test = train_test_split(data['text'],
                                                    data['airline_sentiment'],
                                                    test_size=0.2,
                                                    stratify=data['airline_sentiment'],
                                                    random_state=42)

print(len(y_train[y_train == 0]))
print(len(y_train[y_train == 1]))
print(len(y_train[y_train == 2]))
print(len(y_test[y_test == 0]))
print(len(y_test[y_test == 1]))
print(len(y_test[y_test == 2]))

# dummy_labels = pd.get_dummies(data['airline_sentiment'], prefix_sep='__')
#
# X_train, X_test, y_train, y_test = multilabel_train_test_split(data['text'],
#                                                                dummy_labels,
#                                                                0.2,
#                                                                min_count=3,
#                                                                seed=43)
#
# print(len(y_train[y_train['negative'] == 1]))
# print(len(y_train[y_train['positive'] == 1]))
# print(len(y_train[y_train['neutral'] == 1]))
# print(len(y_test[y_test['negative'] == 1]))
# print(len(y_test[y_test['positive'] == 1]))
# print(len(y_test[y_test['neutral'] == 1]))
#
chi_k = 300
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
TOKENS_ALPHANUMERIC2 = '[A-Za-z0-9]+'

# cv = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC2)
# count_vector = cv.fit_transform(data['text'])
# print(cv.vocabulary_)

# create the pipeline object
pl = Pipeline([
    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                     stop_words=get_stop_words('english'),
                                     norm=None, binary=False, alternate_sign=False,
                                     tokenizer=LemmaTokenizer(),
                                     ngram_range=(1, 2))),
    ('dim_red', SelectKBest(chi2, chi_k)),
    ('int', SparseInteractions(degree=2)),
    ('scale', MaxAbsScaler()),
    ('clf', LogisticRegression(max_iter=10000))
])

# fit the pipeline to our training data
pl.fit(X_train, y_train)
print('a')
pl.score(X_test, y_test)

prediction = pl.predict(X_test)
print(accuracy_score(prediction, y_test))
print(confusion_matrix(prediction, y_test, labels=[0, 1, 2]))
