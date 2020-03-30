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

with open('../data/pickles/binned_df', 'rb') as read_file:
    X_train, y_train = pickle.load(read_file)

lr = LogisticRegression()
cv_lr = cross_validate(lr, X_train, y_train, scoring=['accuracy', 'f1'])
lr.fit(X_train, y_train)

coef = [(coef, feature) for coef, feature in zip(lr.coef_[0], X_train.columns)]

param_dict = {'penalty': ['l1', 'l2'],
              'C':[0.001,.009,0.01,.09,1,5,10,25]}
gs_log = GridSearchCV(lr, param_grid=param_dict, scoring='f1', return_train_score=True)
gs_log.fit(X_train, y_train)

"""
Best_estimator = 
LogisticRegression(C=25, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l1',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)
"""

knn = KNeighborsClassifier()
cv_knn = cross_validate(knn, X_train, y_train, scoring=['accuracy', 'f1'])
'''
 'test_accuracy': array([0.63335895, 0.64206642, 0.63468635]),
 'test_f1': array([0.39604963, 0.38445267, 0.39634146])}
'''


rf = RandomForestClassifier()
cv_lr = cross_validate(rf, X_train, y_train, scoring=['accuracy', 'f1'])
'''
 'test_accuracy': array([0.6368947 , 0.64083641, 0.63637761]),
 'test_f1': array([0.38425443, 0.38102809, 0.3745041 ])}
'''

rf = RandomForestClassifier()
param_dict = {'n_estimators': [5,10,100],
              'max_features': [2,5,10, 15]}
gs_rf = GridSearchCV(rf, param_grid=param_dict, scoring='f1', return_train_score=True)
gs_rf.fit(X_train, y_train)
'''
Best Estimator
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features=10, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=5,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

best_f1 = .39
'''
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

xgb = XGBClassifier()
cv_xg = cross_validate(xgb, X_train, y_train, scoring=['accuracy', 'f1'])

'''
 'test_accuracy': array([0.6727133 , 0.67297048, 0.67420049]),
 'test_f1': array([0.15816528, 0.20663931, 0.2000755 ])}
'''

svc = SVC()
cv_svc = cross_validate(svc, X_train, y_train, scoring=['accuracy', 'f1'])

'''
 'test_accuracy': array([0.667794  , 0.66774293, 0.66774293]),
 'test_f1': array([0., 0., 0.])}
'''
