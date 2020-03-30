
conv_types = ['Aggravated Misdemeanor',
 'B Felony',
 'C Felony',
 'D Felony',
 'Felony - Enhancement to Original Penalty']

def conviction_bin(c_type):

    if  c_type in conv_types:
        return c_type
    else:
        return 'other'
