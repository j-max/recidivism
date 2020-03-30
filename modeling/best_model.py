import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import cross_validate, KFold
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from imblearn.pipeline import Pipeline, make_pipeline


with open('model_objects/best_estimator', 'rb') as read_file:
    lr = pickle.load(read_file)

log_r = lr[1]
with open('../data/pickles/smaller_df', 'rb') as read_file:
    X_train, y_train = pickle.load(read_file)

os = SMOTE()
X_train_res, y_train_res = os.fit_resample(X_train, y_train)

log_r.fit(X_train_res, y_train_res)
y_hat_train = log_r.predict(X_train)
print(recall_score(y_train, y_hat_train))

log_coef = [(coef, feature) for coef, feature in zip(log_r.coef_[0], X_train.columns)]

with open('../data/pickles/smaller_df_test', 'rb') as read_file:
    X_test, y_test = pickle.load(read_file)

y_hat_test = log_r.predict(X_test)
print(recall_score(y_test, y_hat_test))

labels = ['no_recid', 'recid']
cm = confusion_matrix(y_test, y_hat_test)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of Logistic Classifier Trained with SMOTE')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('../figures/models/confusion_final.svg')
