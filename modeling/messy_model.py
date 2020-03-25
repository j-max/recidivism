import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier


with open('../data/pickles/so_many_binaries', 'rb') as read_file:
    X_train, y_train = pickle.load(read_file)

lr = LogisticRegression()
cv = cross_validate(lr, X_train, y_train, cv = 4,
                    scoring=('accuracy', 'f1', 'recall', 'precision', 'roc_auc'),
                    return_train_score=True)

mnb = MultinomialNB()
cv_mnb = cross_validate(mnb, X_train, y_train, cv = 4,
                    scoring=('accuracy', 'f1', 'recall', 'precision', 'roc_auc'),
                    return_train_score=True)


bnb = BernoulliNB()
cv_bnb = cross_validate(bnb, X_train, y_train, cv = 4,
                    scoring=('accuracy', 'f1', 'recall', 'precision', 'roc_auc'),
                    return_train_score=True)




rf = RandomForestClassifier()
cv_rf = cross_validate(rf, X_train, y_train, cv = 4,
                    scoring=('accuracy', 'f1', 'recall', 'precision', 'roc_auc'),
                    return_train_score=True)

rf.fit(X_train, y_train)
print(rf.score(X_train, y_train))

