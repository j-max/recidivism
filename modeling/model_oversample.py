import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import cross_validate, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, accuracy_score
from imblearn.pipeline import Pipeline, make_pipeline

with open('../data/pickles/smaller_df', 'rb') as read_file:
    X_train, y_train = pickle.load(read_file)

lr = LogisticRegression()
cv_lr = cross_validate(lr, X_train, y_train, scoring=['accuracy', 'f1'])
lr.fit(X_train, y_train)

coef = [(coef, feature) for coef, feature in zip(lr.coef_[0], X_train.columns)]

kf = KFold(n_splits=4)

f1_scores = []
accuracy_scores = []
for train_ind, val_ind in kf.split(X_train,y_train):
    X_tt, X_val = X_train.iloc[train_ind], X_train.iloc[val_ind]
    y_tt, y_val = y_train.iloc[train_ind], y_train.iloc[val_ind]
    os = SMOTE()
    X_tt, y_tt = os.fit_resample(X_tt, y_tt)

    lr = LogisticRegression()
    lr.fit(X_tt, y_tt)
    y_hat = lr.predict(X_val)
    f1_scores.append(f1_score(y_val, y_hat))
    accuracy_scores.append(accuracy_score(y_val, y_hat))

model = Pipeline([
        ('sampling', SMOTE()),
        ('classification', LogisticRegression())
    ])

params = {'classification__penalty':['l1', 'l2'],
          'classification__C': np.logspace(-3,3,7)}
grid = GridSearchCV(model, param_grid=params, scoring='recall')
grid.fit(X_train, y_train)
 
'best_f1: .4816'
'best_recall .489'
'''
Best_estimator
Pipeline(memory=None,
         steps=[('sampling',
                 SMOTE(k_neighbors=5, kind='deprecated',
                       m_neighbors='deprecated', n_jobs=1,
                       out_step='deprecated', random_state=None, ratio=None,
                       sampling_strategy='auto', svm_estimator='deprecated')),
                ('classification',
                 LogisticRegression(C=0.1, class_weight=None, dual=False,
                                    fit_intercept=True, intercept_scaling=1,
                                    l1_ratio=None, max_iter=100,
                                    multi_class='warn', n_jobs=None,
                                    penalty='l1', random_state=None,
                                    solver='warn', tol=0.0001, verbose=0,
                                    warm_start=False))],
         verbose=False)
'''


model = Pipeline([
        ('sampling', RandomUnderSampler()),
        ('classification', LogisticRegression())
    ])

params = {'classification__penalty':['l1', 'l2'],
          'classification__C': np.logspace(-3,3,7)}
grid = GridSearchCV(model, param_grid=params, scoring='recall' )
grid.fit(X_train, y_train)
print(f"Logistic recall: {grid.best_score_}")
y_hat = grid.best_estimator_.predict(X_train)
print(f1_score(y_train, y_hat))

with open('model_objects/best_estimator', 'wb') as write_file:
    pickle.dump(grid.best_estimator_, write_file)

'''
f1_best = 0.468
recall_best = .633

Best Estimator 
Pipeline(memory=None,
         steps=[('sampling',
                 RandomUnderSampler(random_state=None, ratio=None,
                                    replacement=False, return_indices=False,
                                    sampling_strategy='auto')),
                ('classification',
                 LogisticRegression(C=0.001, class_weight=None, dual=False,
                                    fit_intercept=True, intercept_scaling=1,
                                    l1_ratio=None, max_iter=100,
                                    multi_class='warn', n_jobs=None,
                                    penalty='l2', random_state=None,
                                    solver='warn', tol=0.0001, verbose=0,
                                    warm_start=False))],
         verbose=False)
'''
 
model = Pipeline([
        ('sampling', SMOTE()),
        ('class', RandomForestClassifier())
    ])
param_dict = {'class__n_estimators': [5,10,100],
              'class__max_features': [2,5,10, 15]}
grid = GridSearchCV(model, param_grid=param_dict, scoring='recall')
grid.fit(X_train, y_train)
print(f"Random Forest Recall: {grid.best_score_}")

'''
 best f1 score 0.47011867093974
 best recall: .5803
'''

model = Pipeline([
        ('sampling', SMOTE()),
        ('class', MultinomialNB())
    ])
param_dict = {'class__alpha': [0,1,2]}

grid = GridSearchCV(model, param_grid=param_dict, scoring='recall')
grid.fit(X_train, y_train)
print(f"MNB recall: {grid.best_score_}")

model = Pipeline([
        ('sampling', SMOTE()),
        ('class', MultinomialNB())
    ])
param_dict = {'class__alpha': [0,1,2]}

grid = GridSearchCV(model, param_grid=param_dict, scoring='precision')
grid.fit(X_train, y_train)
print(f"MNB precision: {grid.best_score_}")


'''
best_f1: 0.474797909364902
best_recall: .6298
'''
