import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

df = pd.read_csv('iowa_recidivism_2019.csv')
y = df['Return to Prison']
# transform target to binary with 1 equal to recidivism
lb_target = LabelBinarizer()
y = pd.Series(lb_target.fit_transform(y).reshape(-1,))

y.index = df.index
X = df.drop('Return to Prison', axis=1)

# for this analysis, I am uninterested in post recidivism data
post_recid_columns = ["Recidivism Type",
                        "New Offense Classification",
                        "New Offense Type",
                        "New Offense Sub Type",
                        "Days to Return",
                        "Target Population",
                      "Recidivism Reporting Year"]

for feature in post_recid_columns:
    X.drop(feature, axis=1, inplace=True)


# train: 19515
# test: 6505 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.25)

def oh_encode(X_train, feature):

    """
    One hot encode a variable in in the training set
    """

    if X_train[feature].isna().sum() != 0:
         print("Remove na values")
         return None, None
    encoder = OneHotEncoder()
    encoded_df = pd.DataFrame(encoder.fit_transform(X_train[[feature]]).toarray())
    # Because there can be many features, include the feature name in the new column names
    enc_cols = [cat+feature for cat in encoder.categories_[0] ]
    encoded_df.columns = enc_cols
    encoded_df.index = X_train.index

    X_train = X_train.drop(feature, axis=1)
    X_train = X_train.join(encoded_df)

    return [X_train, encoder]


encoders = {}
for feature in list(X_train.select_dtypes(include='object')):

    if X_train[feature].isna().sum() < 10:
        X_train.dropna(subset=[feature], inplace=True)
        y_train = y_train[X_train.index]

    else:
         X_train[feature].fillna('no_record', inplace=True)

    if len(X_train[feature].value_counts()) > 2:
       
        X_train, encoder = oh_encode(X_train, feature)
        encoders[feature] = encoder

    else:
        sex_lb = LabelEncoder() 
        sex_lb.fit(['Male', 'Female'])
        X_train['Male'] = sex_lb.transform(X_train[feature])
        X_train.drop(feature, axis=1, inplace=True)

        encoders[feature] = sex_lb

with open('pickles/so_many_binaries', 'wb') as write_file:
    pickle.dump([X_train, y_train], write_file)
