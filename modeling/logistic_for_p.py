import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from statsmodels.discrete.discrete_model import Logit
from statsmodels.formula.api import logit
from sklearn.preprocessing import LabelBinarizer
import sys
sys.path.append('../data')
from race_binner import race_bin


raw_df = pd.read_csv('../data/iowa_recidivism_2019.csv')

with open('../data/pickles/so_many_binaries', 'rb') as read_file:
    data = pickle.load(read_file)


df = data[0]
df['recid']= data[1]

race_cols = []

for col in df.columns:
    if 'Ethnicity' in col:
        race_cols.append(col)

age_cols = []

for col in df.columns:
    if 'Age' in col:
        age_cols.append(col)


formula = 'recid ~ ' + " + ".join(age_cols)

model = Logit(df['recid'], df[age_cols].drop('Under 25Age At Release ', axis=1))
result = model.fit()

# Looking at the P_values, many of the race categories are greater than .05.
model = Logit(df['recid'], df[race_cols].drop( 'White - Non-HispanicRace - Ethnicity', axis=1))
result = model.fit()

# Let's try binning
raw_df['race_bin']  = raw_df['Race - Ethnicity'].apply(race_bin)

formula = 'Return to Prison ~ c(race_bin)'

dummie_race = pd.get_dummies(raw_df['race_bin'], drop_first=False)
dummie_race.drop('hispanic', axis=1, inplace=True)
dummie_race['recid'] = raw_df['Return to Prison']


# From the statsmodels output, hispanics have a relatively low odds of recidivism
# when compared to black, white, and other.
# of the others, white has the highest odds of returning to prison.
recid_lb = LabelBinarizer()
dummie_race['recid'] = recid_lb.fit_transform(dummie_race['recid'])
dummie_race['intercept'] = 1
model = Logit(dummie_race['recid'], dummie_race.drop('recid', axis=1))

result=model.fit()
