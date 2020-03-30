import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import pickle
import seaborn as sns
import sys
sys.path.append('../data')
from race_binner import race_bin
from binner_conviction import conviction_bin
from binner_subtype import subtype_binner

df = pd.read_csv('../data/iowa_recidivism_2019.csv')
df_recid = df[df['Return to Prison'] == "Yes"]
df_no_recid = df[df['Return to Prison'] == "No"]
df['Return to Prison'].hist()

female = df[df.Sex == 'Female']
male = df[df.Sex == 'Male']

'''
# proportionally there is greater recidivism for men: 2.417 to 1.944
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.bar(x=['No', 'Yes'], height=female['Return to Prison'].value_counts(), label = 'Female')
ax2.bar(x=['No', 'Yes'], height=male['Return to Prison'].value_counts(), alpha=.5, label='Male' )
fig.legend()
'''

f_no, f_yes = female['Return to Prison'].value_counts()
print(f"Female ratio of no recidivism to recidvism: {f_no/f_yes}")

m_no, m_yes = male['Return to Prison'].value_counts()
print(f"male ratio of no recidivism to recidvism: {m_no/m_yes}")

with open('../data/pickles/so_many_binaries', 'rb') as read_file:
    data = pickle.load(read_file)

df_binary = data[0]
df_binary['recidivism'] = data[1]

########## AGE EDA#################
ages, age_counts = np.unique(df['Age At Release '].dropna(), return_counts=True)

nr_ages, nr_age_counts = np.unique(df_no_recid['Age At Release '].dropna(), return_counts=True)

r_ages, r_age_counts = np.unique(df_recid['Age At Release '].dropna(), return_counts=True)

age_order = ['Under 25', '25-34', '35-44', '45-54', '55 and Older']
age_counts = df['Age At Release '].value_counts()
nr_age_counts = df_no_recid['Age At Release '].value_counts()
r_age_counts = df_recid['Age At Release '].value_counts()

fig, (ax1, ax2) = plt.subplots(2,1)
sns.barplot(age_order, age_counts.loc[age_order], ax=ax1, color = 'purple')
sns.barplot(age_order, nr_age_counts.loc[age_order], ax=ax2, alpha=.5, color='blue', label='No recidivism')
ax1.title.set_text('Total Recidivism by Age Group') 
sns.barplot(age_order, r_age_counts.loc[age_order], ax=ax2, alpha=.5, color='red', label='Recidivism')
ax2.title.set_text('Recidivism vs Non-recidivism by Age Group')
ax2.legend()
plt.tight_layout()
plt.savefig('../figures/eda/age.svg')

########## Race EDA ##########
race_counts = df['Race - Ethnicity'].value_counts()
fig, ax1 = plt.subplots()

sns.barplot(race_counts.index, race_counts, ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
plt.tight_layout()

print(df['Race - Ethnicity'].value_counts(normalize=True))

df['race_bin'] = df['Race - Ethnicity'].apply(race_bin)

height = df['race_bin'].value_counts()
x = list(height.index)
height = list(height)
fig, ax1 = plt.subplots()
sns.barplot(y = height,  x = x, ax=ax1)
plt.show()


race_recid = df.groupby('race_bin')['Return to Prison'].value_counts(normalize=True)

race_groups = [('black', 'Yes'), ('hispanic', 'Yes'),
               ('other', 'Yes'),  ('white', 'Yes')]
races= ['black', 'hispanic', 'other', 'white']
fig, ax1 = plt.subplots()
sns.barplot(x=races, y = race_recid.loc[race_groups].values, ax=ax1, palette=['blue', 'green', 'blue','blue'])
plt.title('Hispanic Recidivism Rates\n Considerably Lower than Other Groups')
plt.ylabel('Recidivism Rate')
plt.savefig('../figures/eda/race_binned.svg')


############ Conviction EDA ###########
conv_class = df.groupby('Offense Classification')['Return to Prison'].value_counts()
conv_r_yes = conv_class.loc[:,'Yes']

norm_conv_class = df.groupby('Offense Classification')['Return to Prison'].value_counts(normalize=True)

norm_conv_r_yes = norm_conv_class.loc[:,'Yes']

fig, (ax1,ax2) = plt.subplots(1,2)
sns.barplot(x=conv_r_yes.index, y = conv_r_yes, ax=ax1 )
sns.barplot(x=norm_conv_r_yes.index, y = norm_conv_r_yes, ax=ax2 )
plt.ylabel('Recidivism Rate')
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('../figures/eda/conv_class.svg')


df['conv_bin'] = df['Offense Classification'].apply(conviction_bin)
norm_conv_bin = df.groupby('conv_bin')['Return to Prison'].value_counts(normalize=True)

norm_conv_bin_yes = norm_conv_bin.loc[:,'Yes']


fig, ax1 = plt.subplots()
sns.barplot(x=norm_conv_bin_yes.index, y = norm_conv_bin_yes, ax=ax1 )
plt.ylabel('Recidivism Rate')
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('../figures/eda/conv_bin.svg')

########### Offense type EDA ###########

fig, ax1 = plt.subplots()
offense_count = df['Offense Type'].value_counts()
sns.barplot(x=offense_count.index, y=offense_count)
plt.title('Counts of Different Offense Types')
plt.savefig('../figures/eda/offense_type.svg')

offense_gb = df.groupby('Offense Type')['Return to Prison'].value_counts()

offense_recid = offense_gb[:,'Yes']

fig, ax1 = plt.subplots()
sns.barplot(x=offense_recid.index, y=offense_recid)
plt.title('Counts of Different Offense Types')
plt.savefig('../figures/eda/offense_type_recid.svg')

offense_gb_norm = df.groupby('Offense Type')['Return to Prison'].value_counts(normalize=True)

offense_recid_norm = offense_gb_norm[:,'Yes']

fig, ax1 = plt.subplots()
sns.barplot(x=offense_recid_norm.index, y=offense_recid_norm)
plt.title('Counts of Different Offense Types')
plt.savefig('../figures/eda/offense_type_recid_norm.svg')

########## EDA offense subtype ##########

fig, ax1 = plt.subplots()
off_subt_count = df['Offense Subtype'].value_counts()
sns.barplot(x=off_subt_count.index, y=off_subt_count, ax=ax1)
plt.title('Counts of Different Offense Subtypes Types')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('../figures/eda/off_subt_type.svg')


off_subt_norm= df.groupby('Offense Subtype')['Return to Prison'].value_counts(normalize=True, ascending=True)

offense_subt_recid_norm = off_subt_norm[:,'Yes']

fig, ax1 = plt.subplots()
sns.barplot(x=offense_subt_recid_norm.index, y=offense_subt_recid_norm)
plt.title('Counts of Different Offense Subtypes')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('../figures/eda/offense_subtype_recid_norm.svg')


df['subt_bin'] = df['Offense Subtype'].apply(subtype_binner)

off_subt_bin_norm= df.groupby('subt_bin')['Return to Prison'].value_counts(normalize=True, ascending=True)

off_subt_bin_norm = off_subt_bin_norm[:,'Yes']


fig, ax1 = plt.subplots()
sns.barplot(x=off_subt_bin_norm.index, y=off_subt_bin_norm)
plt.title('Counts of Different Offense Subtypes\n with Binning')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('../figures/eda/offense_subtype_bin_norm.svg')

