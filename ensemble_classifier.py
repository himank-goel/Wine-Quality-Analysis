import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter

# Function for accuracy


def accuracy(Y_test, Y_pred):
    equal = 0
    for i in xrange(len(Y_pred)):
        if Y_pred[i] == Y_test[i]:
            equal += 1
        # print Y_test[i]

    print 'Accuracy = %s' % (float(equal)/len(Y_pred))


# Loading the dataset
# dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
dataset_url = 'winequality\winequality-white.csv'
data = pd.read_csv(dataset_url, sep=';')

# Splitting into training and testing datasets
Y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y, test_size=0.25, random_state=123, stratify=Y)


# Setting the Classifiers
classifiers = [RandomForestClassifier(n_estimators=10, criterion='gini'), RandomForestClassifier(n_estimators=10, criterion='entropy'), ExtraTreesClassifier(
    n_estimators=10, criterion='gini'), ExtraTreesClassifier(n_estimators=10, criterion='entropy'), GradientBoostingClassifier(n_estimators=10)]

Y_pred = [[] for i in xrange(len(classifiers))]

#Training and prediction
for i, classifier in enumerate(classifiers):
    classifier.fit(X_train, Y_train)
    Y_pred[i] = classifier.predict(X_test)

Y_mean = []

for i in xrange(len(Y_pred[0])):
    counter = Counter([Y_pred[classifier_id][i]
                       for classifier_id in xrange(len(classifiers))])
    vote = counter.most_common(1)[0][0]
    Y_mean.append(vote)

# Calculating Accuracy
accuracy(Y_test.tolist(), Y_mean)