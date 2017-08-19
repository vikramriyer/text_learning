'''
Formula for Naive Bayes
P(A|B) = (P(B|A)*P(A))/P(B)
where,
P(A|B) is called posterior probability, since it depends upon B
P(A) is called prior or marginal probability. It is prior in the sense that it does not take any account of B
P(B) is called prior marginality of B as it acts as a normalizing constant
P(B|A) is called likelihood or simply prob of B given A

So the above formula can verbosely also be written as:
posterior = likelihood*prior/normalizing constant

simple example : king
complex example: liver disorder

explain laplace smoothing for zero frequency condition
'''
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SklearnNaiveBayes(object):

  def __init__(self):
    pass

  def use_naive_bayes(self, X_train, X_test, y_train, y_test):
    
    # let's split up the dataset into training and testing set
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)

    '''
    print "Predicted, Actual"
    for x,y in zip(y_predict, y_test):
      print "\t%s\t\t %s" % (x,y[0])
    print "Accuracy: %s percent" % int(accuracy_score(y_predict, y_test)*100)
    '''
    return y_predict

class VikramGaussianNB(object):
  def __init__(self):
    pass

  def fit(self, X, y):
    separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
    self.model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)] for i in separated])
    return self

  def _prob(self, x, mean, std):
    exponent = np.exp(- ((x - mean)**2 / (2 * std**2)))
    return np.log(exponent / (np.sqrt(2 * np.pi) * std))

  def predict_log_proba(self, X):
    return [[sum(self._prob(i, *s) for s, i in zip(summaries, x)) for summaries in self.model] for x in X]

  def predict(self, X):
    return np.argmax(self.predict_log_proba(X), axis=1)

  def score(self, X, y):
    return sum(self.predict(X) == y) / len(y)

  def accuracy(self, prediction_set, test_set):
    return np.sum(prediction_set.reshape((len(prediction_set),1)) == test_set)

def prepare_data(X, y):
  
  # we have the movie_category as categorical variables here, as we have 3 categories to choose from
  # hence we can encode it with one hot encoder
  encoder = LabelEncoder()
  encoder.fit(X)
  X = pd.Series(encoder.transform(X))
  X = X.reshape(14,1)
  y = y.reshape(14,1)

  return X, y

def main():

  # importing data, I have created a toy dataset, which we will learn from and later use the amazon food reviews dataset
  #X = pd.read_csv('movies.csv')
  X = pd.read_csv('prima.csv')
  #y = X.Watch
  y = X.test_result
  #X.drop('Watch', inplace=True, axis=1)
  X.drop('test_result', inplace=True, axis=1)

  # step 1: encode to convert categorical variable into numbers
  #X, y = prepare_data(X, y)

  # step 2: split data into train and test
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)


  # step 3: use sklearn naive bayes
  sklearn_nb = SklearnNaiveBayes()
  pred_naive_bayes = sklearn_nb.use_naive_bayes(X_train, X_test, y_train, y_test)

  # step 4: simulate sklearn with own implementation of Naive Bayes
  clf = VikramGaussianNB()
  clf = clf.fit(np.array(X_train).astype(float), np.array(y_train).astype(float))
  pred_vikram_naive_bayes = clf.predict(np.array(X_test).astype(float))

  print "Naive Bayes Accuracy: %s percent" % int(accuracy_score(pred_naive_bayes, y_test)*100)
  print "Vikram Naive Bayes Accuracy: %s percent" % int(accuracy_score(pred_vikram_naive_bayes, y_test)*100)

if __name__ == "__main__":
  main()