import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

with open('../data/pickles/so_many_binaries', 'rb') as read_file:
    X_train, y_train = pickle.load(read_file)

lr = LogisticRegression()
cv_lr = cross_validate(lr, X_train, y_train, cv = 4,
                    scoring=('accuracy', 'f1', 'recall', 'precision', 'roc_auc'),
                    return_train_score=True)

lr.fit(X_train, y_train)
lr_coef_df = pd.DataFrame(lr.coef_, columns=X_train.columns).T
'''
There is no doubt heavy collinearity here, but the coefficients give
insight into what may be good places to continue eda. 
bottom 5 for coeficient importances with all variables dummied:
no_recordMain Supervising District                 -1.089151
55 and OlderAge At Release                         -0.615058
Interstate CompactMain Supervising District        -0.500984
ISCMain Supervising District                       -0.468404
White - HispanicRace - Ethnicity                   -0.385191

top 5:
25-34Age At Release                                 0.330808
4JDMain Supervising District                        0.343059
American Indian or Alaska Native - Non-Hispanic...  0.349490
Male                                                0.408802
5JDMain Supervising District                        0.484443
'''

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
# f1 test score of .37. Not so hot, but much better than logistic

rf.fit(X_train, y_train)

svc = SVC(max_iter=300)
cv_svc = cross_validate(svc, X_train, y_train, cv = 4,
                        scoring=('accuracy', 'f1', 'recall', 'precision', 'roc_auc'),
                    return_train_score=True)
# vanilla svc works pretty well
