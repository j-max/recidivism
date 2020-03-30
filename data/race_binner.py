
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

