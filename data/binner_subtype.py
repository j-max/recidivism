
less_than_1_per = [
    'Arson',
    'Sex Offender Registry/Residency',
    'Flight/Escape',
    'Kidnap',
    'Special Sentence Revocation',
    'Prostitution/Pimping',
    'Stolen Property',
    'Animals']

def subtype_binner(subt):

    if subt in less_than_1_per:
        return 'other'
    else:
        return subt
