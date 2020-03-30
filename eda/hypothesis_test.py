import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import pickle

plt.ioff()
''' These hypothesis tests are conducted exclusively on the training set,
so as not to bias the model with knowledge of the test'''

# df encoded has every feature ecoded into a binary through simple binarization and one hot encoding
# see functions in data/encoding_funcs.py
with open('../data/pickles/so_many_binaries', 'rb') as open_file:
    df = pickle.load(open_file)

df_encoded = df[0]
df_encoded['recidivism'] = df[1] 

female = df_encoded[df_encoded.Male == 0]
male = df_encoded[df_encoded.Male == 1]

with open('../data/pickles/pre_encoding', 'rb') as open_file:
    df_pre = pickle.load(open_file)

df_pre_encoded = df_pre[0]
df_pre_encoded['recidivism'] = df_pre[1] 

female = df_encoded[df_encoded.Male == 0]
male = df_encoded[df_encoded.Male == 1]

# proportionally there is greater recidivism for men: 2.417 to 1.944
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.bar(x=['No_recid', 'Yes_recid'], height=female['recidivism'].value_counts(), label = 'Female')
ax2.bar(x=['No_recid', 'Yes_recid'], height=male['recidivism'].value_counts(), alpha=.5, label='Male' )
fig.legend(bbox_to_anchor=(0.,0,  1., .102))
fig.suptitle('Women less likely to be reincarcerated than men')
plt.savefig('../figures/eda/sex.svg')
f_no, f_yes = female['recidivism'].value_counts()
print(f"Female ratio of no recidivism to recidvism: {f_no/f_yes}")

m_no, m_yes = male['recidivism'].value_counts()
print(f"male ratio of no recidivism to recidvism: {m_no/m_yes}")

#chi_squared sex test
# Null: being male and undergoing recidivism are independent of one another 
# Alt: being male and recidivism are dependent on one another 
# p-value .05
crosstab = pd.crosstab(df_encoded['Male'], df_encoded['recidivism'])
print("Chi-squared Male/Recidivsm test ")
print(chi2_contingency(crosstab))
# chi2 confirms there is significant dependence of being male and being reincarcerated.

# Null: There is no dependence of age on reincarceration
# Alt: Age and reincarceration are dependent
crosstab = pd.crosstab(df_pre_encoded['Age At Release '], df_pre_encoded['recidivism'])
print(df_pre_encoded['Age At Release '].unique())
print(chi2_contingency(crosstab))
 
#chi_race test
crosstab = pd.crosstab(df_pre_encoded['Race - Ethnicity'], df_pre_encoded['recidivism'])
#print(chi2_contingency(crosstab))

ethnicity_list = []

for column in df_encoded.columns:
    if 'Ethnicity' in column:
        ethnicity_list.append(column)


# Null: Ethnicity and race are independent
# Alt: Ethnicity and race are dependent
for ethnicity in ethnicity_list:
    crosstab = pd.crosstab(df_encoded[ethnicity],
                        df_encoded['recidivism'])
    #print(ethnicity)
    #print(chi2_contingency(crosstab))


    if chi2_contingency(crosstab)[1] < .05:
        print(ethnicity)

# the only racial groups that pass the test on their own are:
# American Indian or Alaska Native - Non-HispanicRace - Ethnicity
# Asian or Pacific Islander - Non-HispanicRace - Ethnicity
# White - HispanicRace - Ethnicity
# White - Non-HispanicRace - Ethnicity
# no_recordRace - Ethnicity 


# chi_offense
crosstab = pd.crosstab(df_pre_encoded['Offense Subtype'], df_pre_encoded['recidivism'])
# print(chi2_contingency(crosstab))

crosstab = pd.crosstab(df_pre_encoded['Release Type'], df_pre_encoded['recidivism'])
# print(chi2_contingency(crosstab))
 
