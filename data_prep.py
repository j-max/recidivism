import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle
import sys
sys.path.append('../')

# there are 26020 records in the raw data
df = pd.read_csv('data/iowa_recidivism_2019.csv')

#target is a binary
# 8681 yes
# 17339 no
# 2/1: no/yes


y = df['Return to Prison']

# transform target to binary with 1 equal to recidivism
lb_target = LabelBinarizer()
y = pd.Series(lb_target.fit_transform(y).reshape(-1,))

y.index = df.index
X = df.drop('Return to Prison', axis=1)

# train: 19515
# test: 6505 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.25)

# split performed before EDA so as to not bias test set in any way

####### feature preprocessing #####
# Sex: Male: 17012, Female: 2501
# two sex na's to drop
X_train.dropna(axis = 0, subset=['Sex'], inplace=True)

# convert to male 1, female 0
sex_lb = LabelEncoder() 
sex_lb.fit(['Male', 'Female'])
X_train['male'] = sex_lb.transform(X_train.Sex)
X_train.drop('Sex', axis=1, inplace=True)

# race
# In order to make the results more interpretable, I bin race according to the following:

race_dict = {'white': ['White - Non-Hispanic', 'White -'],
             'black': ['Black - Non-Hispanic', 'Black -'],
             'hispanic': ['White - Hispanic', 'Black - Hispanic',
                          'American Indian or Alaska Native - Hispanic',
                          'Asian or Pacific Islander - Hispanic'],
             'other': ['American Indian or Alaska Native - Non-Hispanic',
                       'Asian or Pacific Islander - Non-Hispanic', 'N/A -']
             }

def race_bin(race):
    """
    Consolidate racial demographics into White, Black, Hispanic, and Other
    """
    for key, value in race_dict.items():
        if race in value:
            return key
    else:
        return 'other'

X_train['race_binned'] = X_train['Race - Ethnicity'].apply(race_bin)
X_train.drop('Race - Ethnicity', axis=1, inplace=True)

# After binning there are:
# white: 13221, black: 4588, hispanic: 1179, other: 501

le_race = LabelEncoder()
le_race.fit(['white', 'black', 'hispanic', 'other'])
race = le_race.transform(X_train.race_binned)
ohe_race= OneHotEncoder(sparse=False)
oh_race = pd.DataFrame(ohe_race.fit_transform(race.reshape(-1, 1)))
oh_race.columns = le_race.classes_
oh_race.index = X_train.index
X_train = X_train.join(oh_race)
X_train.drop('race_binned', axis=1, inplace=True)

# Age at release
'''
25-34           7179
35-44           4647
Under 25        3425
45-54           3298
55 and Older     964
'''

# creating instance of one-hot-encoder
age_enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(age_enc.fit_transform(X_train[['Age At Release ']]).toarray())
enc_df.columns = age_enc.categories_[0]
enc_df.index = X_train.index
X_train = X_train.join(enc_df)
X_train.drop('Age At Release ', axis=1)


offense_enc = OneHotEncoder(handle_unknown='ignore')
offense_df = pd.DataFrame(offense_enc.fit_transform(X_train[['Offense Subtype']]).toarray())
offense_df.columns = offense_enc.categories_[0]
offense_df.index = X_train.index
X_train = X_train.join(offense_df)
X_train.drop('Offense Subtype', axis=1)




######### Export for messy_model.py######
'''
features_for_messy_model = []
features_for_messy_model.extend(offense_enc.categories_[0].tolist())
features_for_messy_model.extend(age_enc.categories_[0].tolist())
features_for_messy_model.extend(['male'])

X_train_for_messy = X_train[features_for_messy_model]
y_train_for_messy = y_train[X_train.index]

with open('data/pickles/messy_df', 'wb') as write_file:
   pickle.dump([X_train_for_messy, y_train_for_messy], write_file)

'''
class encoded_variable():

    def __init__(self, feature, encoder):

        self.feature= feature 
        self.encoder = encoder

    def encode(self, X_train):

        encoded_df = pd.DataFrame(self.encoder.fit_transform(X_train[[self.feature]]).toarray())
        encoded_df.columns = encoder.categories_[0]
        encoded_df.index = X_train.index

        return encoded_df
