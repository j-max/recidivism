# prison_recidivism

# Guiding questions
1. What are the main factors which drive recidivism in the sample poplution?
2. Does recidivism rate mostly depend on individual characteristics or circumstance?
3. How could fiscal resources be redirected in order to improve recidivism rates?



## Data Source
The data for this project can be found at https://data.iowa.gov/Correctional-System/3-Year-Recidivism-for-Offenders-Released-from-Pris/mw8r-vqy4

## Feature Engineering
   After pulling the raw data from the database, the data is converted to binary features with the code found in the data/feature_end.py file. None of the Iowa data is numeric, not even age, which is provided as a range. Therefore, I created dummy variables out of all features.

   The final model features are: race, age at release, conviction class, conviction type, conviction subtype, release type, supervising district, target population, initial supervision status, recidivism supervision status.

### Notes on Target and  Individual Features
#### Target
  * The dataset is imbalanced. Of the 24,387 records left after reducing the dataset, 8,334 are marked as recidivist and 16,053 are not recidivist.  That is a ratio of 1.93 non-recidivists to each recidivist.  This imbalance will be addressed by random-oversampling in the training sets of the models (see modeling section).
#### race
     ```
	 There are 11 racial categories in the raw dataset:
		 race_American Indian or Alaska Native - Hispanic           20
		 race_American Indian or Alaska Native - Non-Hispanic      502
		 race_Asian or Pacific Islander - Hispanic                   5
		 race_Asian or Pacific Islander - Non-Hispanic             192
		 race_Black -                                                2
		 race_Black - Hispanic                                      37
		 race_Black - Non-Hispanic                                6109
		 race_N/A -                                                  5
		 race_White -                                               12
		 race_White - Hispanic                                    1522
		 race_White - Non-Hispanic                               17584
	 
	 Each record has only one category, so to make interpretation easier, I drop records not in the following  categories:
	 race_American Indian or Alaska Native - Non-Hispanic, 502
	 race_Asian or Pacific Islander - Non-Hispanic, 192
	 race_Black - Non-Hispanic, 6109
	 race_White - Non-Hispanic, 17584
	 
	 I drop these either becasue the count is very small, or the individual identifies as having more than one race.

	 ```
## Modeling
  * Logistic Regression
