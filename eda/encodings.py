import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

df = pd.read_csv('../data/iowa_recidivism_2019.csv')
y = df['Return to Prison']
# transform target to binary with 1 equal to recidivism
lb_target = LabelBinarizer()
y = pd.Series(lb_target.fit_transform(y).reshape(-1,))

y.index = df.index
X = df.drop('Return to Prison', axis=1)

# train: 19515
# test: 6505 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.25)

X_train.dropna(axis = 0, subset=['Sex'], inplace=True)

class encoded_X_train():

    def __init__(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train
        
    def oh_encode(self, feature):

        if self.X_train[feature].isna().sum() != 0:
            drop = input('drop na? y/n')
            if drop == 'n':
                return
            else:
                self.X_train.dropna(subset=feature, axis=1, inplace=True)
                
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoded_df = pd.DataFrame(encoder.fit_transform(self.X_train[[feature]]).toarray())
        # Because there can be many features, include the feature name in the new column names
        enc_cols = [cat+feature for cat in encoder.categories_[0] ]
        encoded_df.columns = enc_cols
        encoded_df.index = self.X_train.index

        self.X_train.drop(feature, axis=1, inplace=True)

        self.X_train = self.X_train.join(encoded_df)

        return encoder

    def binary_encode(self, label_encoder, label_list):

        # convert to male 1, female 0
        lb = LabelEncoder() 
        lb.fit(label_list)
        self.X_train[label_list[0]] = sex_lb.transform(self.X_train[self.feature])
        self.X_train.drop(self.feature, axis=1, inplace=True)

        self.X_train = self.X_train.join(encoded_df)

        return lb


enc_X_train = encoded_X_train(X_train, y_train)
enc_X_train.X_train.dropna(subset=['Age At Release '], inplace=True)
enc_X_train.oh_encode('Age At Release ')

y_train = y_train[enc_X_train.X_train.index]


'''
X_train.dropna(subset=['Age At Release '], inplace=True)
age_encode = encoded_variable('Age At Release ', OneHotEncoder(handle_unknown='ignore'))
y_train = y_train[X_train.index]
age_encode_cols = age_encode.oh_encode(X_train)

offence_enc = encoded_variable('Offense Subtype', OneHotEncoder(handle_unknown='ignore'))
offence_enc_cols = offence_enc.oh_encode(age_encode_cols)

sex_binarized = encoded_variable('Sex',  ['Male', 'Female'])
sex_col = sex_binarized.binary_encode(offence_enc_cols, ['Male', 'Female'])
'''
