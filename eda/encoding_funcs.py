import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle
import sys
sys.path.append('../data')
from race_binner import race_bin
from binner_conviction import conviction_bin
from binner_subtype import subtype_binner


df = pd.read_csv('../data/iowa_recidivism_2019.csv')
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

X_train_bin, X_test_bin, y_train_bin, y_test_bin = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy() 

X_train_less, X_test_less, y_train_less, y_test_less = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy() 


with open('../data/pickles/pre_encoding', 'wb') as write_file:
    pickle.dump([X_train, y_train], write_file)

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
    enc_cols = [str(cat)+' '+ feature for cat in encoder.categories_[0] ]
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

with open('../data/pickles/so_many_binaries', 'wb') as write_file:
    pickle.dump([X_train, y_train], write_file)

########## Binned Categoricals ########## 
# Creating binned dataframes for modeling

X_train_bin['race_binned'] = X_train_bin['Race - Ethnicity'].apply(race_bin)
X_train_bin.drop('Race - Ethnicity', axis=1, inplace=True)

X_train_bin['conv_bin'] = X_train_bin["Offense Classification"]
X_train_bin.drop('Offense Classification', axis=1, inplace=True)

X_train_bin['subtype_bin'] = X_train_bin['Offense Subtype'].apply(subtype_binner)
X_train_bin.drop("Offense Subtype", axis=1, inplace=True )

encoders = {}
for feature in list(X_train_bin.select_dtypes(include='object')):

    if X_train_bin[feature].isna().sum() < 10:
        X_train_bin.dropna(subset=[feature], inplace=True)
        y_train_bin = y_train_bin[X_train_bin.index]

    else:
         X_train_bin[feature].fillna('no_record', inplace=True)

    if len(X_train_bin[feature].value_counts()) > 2:
       
        X_train_bin, encoder = oh_encode(X_train_bin, feature)
        encoders[feature] = encoder

    else:
        sex_lb = LabelEncoder() 
        sex_lb.fit(['Male', 'Female'])
        X_train_bin['Male'] = sex_lb.transform(X_train_bin[feature])
        X_train_bin.drop(feature, axis=1, inplace=True)

        encoders[feature] = sex_lb

X_train_bin, fy_oh = oh_encode(X_train_bin, 'Fiscal Year Released')

with open('../data/pickles/binned_df', 'wb') as write_file:
    pickle.dump([X_train_bin, y_train_bin], write_file)


########## pared down dataframe ########## 

X_train_less['race_binned'] = X_train_less['Race - Ethnicity'].apply(race_bin)
X_train_less.drop('Race - Ethnicity', axis=1, inplace=True)

X_train_less['conv_bin'] = X_train_less["Offense Classification"]
X_train_less.drop('Offense Classification', axis=1, inplace=True)

X_train_less['subtype_bin'] = X_train_less['Offense Subtype'].apply(subtype_binner)
X_train_less.drop("Offense Subtype", axis=1, inplace=True )

X_train_less.drop(['Fiscal Year Released', 'Main Supervising District', 'Release Type'], axis=1, inplace=True)

less_encoders = {}
for feature in list(X_train_less.select_dtypes(include='object')):

    if X_train_less[feature].isna().sum() < 10:
        X_train_less.dropna(subset=[feature], inplace=True)
        y_train_less = y_train_less[X_train_less.index]

    else:
         X_train_less[feature].fillna('no_record', inplace=True)

    if len(X_train_less[feature].value_counts()) > 2:
       
        X_train_less, encoder = oh_encode(X_train_less, feature)
        less_encoders[feature] = encoder

    else:
        sex_lb = LabelEncoder() 
        sex_lb.fit(['Male', 'Female'])
        X_train_less['Male'] = sex_lb.transform(X_train_less[feature])
        X_train_less.drop(feature, axis=1, inplace=True)

        less_encoders[feature] = sex_lb

with open('../data/pickles/smaller_df', 'wb') as write_file:
    pickle.dump([X_train_less, y_train_less], write_file)

########## test prep ########## 

X_test_less['race_binned'] = X_test_less['Race - Ethnicity'].apply(race_bin)
X_test_less.drop('Race - Ethnicity', axis=1, inplace=True)

X_test_less['conv_bin'] = X_test_less["Offense Classification"]
X_test_less.drop('Offense Classification', axis=1, inplace=True)

X_test_less['subtype_bin'] = X_test_less['Offense Subtype'].apply(subtype_binner)
X_test_less.drop("Offense Subtype", axis=1, inplace=True )

X_test_less.drop(['Fiscal Year Released', 'Main Supervising District', 'Release Type'], axis=1, inplace=True)

for feature in list(X_test_less.select_dtypes(include='object')):
    print(feature)
    if X_test_less[feature].isna().sum() < 10:
        X_test_less.dropna(subset=[feature], inplace=True)
        y_test_less = y_test_less[X_test_less.index]

    else:
         X_test_less[feature].fillna('no_record', inplace=True)
         
    if len(X_test_less[feature].value_counts()) > 2:
        encoder = less_encoders[feature]
        encoded_df = pd.DataFrame(encoder.transform(X_test_less[[feature]]).toarray())
        # Because there can be many features, include the feature name in the new column names
        enc_cols = [str(cat)+' '+ feature for cat in encoder.categories_[0] ]
        encoded_df.columns = enc_cols
        encoded_df.index = X_test_less.index

        X_test_less = X_test_less.drop(feature, axis=1)
        X_test_less = X_test_less.join(encoded_df)


    else:
        sex_lb = LabelEncoder() 
        sex_lb.fit(['Male', 'Female'])
        X_test_less['Male'] = sex_lb.transform(X_test_less[feature])
        X_test_less.drop(feature, axis=1, inplace=True)

with open('../data/pickles/smaller_df_test', 'wb') as write_file:
    pickle.dump([X_test_less, y_test_less], write_file)
