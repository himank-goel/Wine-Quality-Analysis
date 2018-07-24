import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Function for accuracy


def accuracy(Y_test, Y_pred):
    equal = 0
    for i in xrange(len(Y_pred)):
        if Y_pred[i] == Y_test[i]:
            equal += 1
        # print Y_test[i]

    print 'Accuracy = %s' % (float(equal)/len(Y_pred))


# Loading the dataset
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

# Splitting into training and testing datasets
Y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y, test_size=0.25, random_state=123, stratify=Y)

# Preprocessing the Data
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestClassifier(n_estimators=500, criterion='entropy'))

# Setting the HyperParameters
hyperparameters = {'randomforestclassifier__max_features': [
    'auto', 'sqrt', 'log2'], 'randomforestclassifier__max_depth': [None, 5, 3, 1, 7]}

# Fitting the classfier
clf = GridSearchCV(pipeline, hyperparameters, cv=5)
clf.fit(X_train, Y_train)

# Making prediction on the test set
Y_pred = clf.predict(X_test)

# Calculating Accuracy
accuracy(Y_test.tolist(), Y_pred)
