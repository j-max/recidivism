import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


with open('../data/pickles/smaller_df', 'rb') as read_file:
    X_train, y_train = pickle.load(read_file)

lr = LogisticRegression()
cv_lr = cross_validate(lr, X_train, y_train, scoring=['accuracy', 'f1'])
lr.fit(X_train, y_train)

coef = [(coef, feature) for coef, feature in zip(lr.coef_[0], X_train.columns)]

param_dict = {'penalty': ['l1', 'l2'],
              'C':[0.001,.009,0.01,.09,1,5,10,25]}
gs_log = GridSearchCV(lr, param_grid=param_dict, scoring='recall', return_train_score=True)
gs_log.fit(X_train, y_train)


knn = KNeighborsClassifier()
cv_knn = cross_validate(knn, X_train, y_train, scoring=['accuracy', 'f1'])


rf = RandomForestClassifier()
cv_rf= cross_validate(rf, X_train, y_train, scoring='accuracy' )
rf = RandomForestClassifier()
param_dict = {'n_estimators': [5,10,100],
              'max_features': [2,5,10, 15]}
gs_rf = GridSearchCV(rf, param_grid=param_dict, scoring='f1', return_train_score=True)
gs_rf.fit(X_train, y_train)

'''
best accuracy : 0.25765853538983996
'''

xgb = XGBClassifier()
xgb.fit(X_train, y_train)

xgb = XGBClassifier()
cv_xg = cross_validate(xgb, X_train, y_train, scoring=['accuracy', 'f1'])


svc = SVC()
cv_svc = cross_validate(svc, X_train, y_train, scoring=['accuracy', 'f1'])


