import numpy as np
import pandas as pd

# --- Existing functions (unchanged) ---

def shock_index(data):
    """ HR/SBP ratio. """
    return data['HR'] / data['SBP']


def age_normalised_shock_index(data):
    """ HR/(SBP * Age). """
    return data['HR'] / (data['SBP'] * data['Age'])


def modfied_shock_index(data):
    """ HR/(MAP * Age). """
    return data['HR'] / (data['MAP'] * data['Age'])


def bun_cr(data):
    """ BUN/Creatinine. """
    return data['BUN'] / data['Creatinine']


def sao2_fio2(data):
    """ SaO2/FiO2. """
    return data['SaO2'] / data['FiO2']


def urea_creatinine(data):
    """ BUN/Creatinine. """
    return data['BUN'] / data['Creatinine']


def pulse_pressure(data):
    """ SBP - DBP. """
    return data['SBP'] - data['DBP']


def cardiac_output(data):
    """ Pulse Pressure * HR. """
    pp = data['SBP'] - data['DBP']
    return pp * data['HR']


# --- Modified partial_sofa for a single row ---
def partial_sofa(data):
    """
    Partial SOFA score calculated from a single row.
    Requires: 'Platelets', 'Bilirubin_total', 'MAP', 'Creatinine'
    """
    # Coagulation (Platelets)
    platelets = data['Platelets']
    if platelets >= 150:
        platelets_score = 0
    elif 100 <= platelets < 150:
        platelets_score = 1
    elif 50 <= platelets < 100:
        platelets_score = 2
    elif 20 <= platelets < 50:
        platelets_score = 3
    else:  # platelets < 20
        platelets_score = 4

    # Liver (Bilirubin_total)
    bilirubin = data['Bilirubin_total']
    if bilirubin < 1.2:
        bilirubin_score = 0
    elif 1.2 <= bilirubin <= 1.9:
        bilirubin_score = 1
    elif 1.9 < bilirubin <= 5.9:
        bilirubin_score = 2
    elif 5.9 < bilirubin <= 11.9:
        bilirubin_score = 3
    else:  # bilirubin > 11.9
        bilirubin_score = 4

    # Cardiovascular (MAP)
    map_value = data['MAP']
    map_score = 0 if map_value >= 70 else 1

    # Renal (Creatinine)
    creatinine = data['Creatinine']
    if creatinine < 1.2:
        creatinine_score = 0
    elif 1.2 <= creatinine <= 1.9:
        creatinine_score = 1
    elif 1.9 < creatinine <= 3.4:
        creatinine_score = 2
    elif 3.4 < creatinine <= 4.9:
        creatinine_score = 3
    else:  # creatinine > 4.9
        creatinine_score = 4

    return platelets_score + bilirubin_score + map_score + creatinine_score


# --- Wrapper function that calls the above functions safely ---
def compute_derived_features(row):
    """
    Given a row (pd.Series) containing clinical data, compute a set of derived features.
    For each feature, if a required key is missing or (for ratios) the denominator is 0,
    the feature value is set to np.nan.
    
    Expected keys:
      'HR', 'SBP', 'Age', 'MAP', 'BUN', 'Creatinine', 'SaO2', 'FiO2',
      'DBP', 'Platelets', 'Bilirubin_total'
    """
    features = {}
    
    # Shock Index: HR/SBP
    if pd.isnull(row.get('HR')) or pd.isnull(row.get('SBP')) or row['SBP'] == 0:
        features['shock_index'] = np.nan
    else:
        features['shock_index'] = shock_index(row)

    # Age Normalised Shock Index: HR/(SBP * Age)
    if (pd.isnull(row.get('HR')) or pd.isnull(row.get('SBP')) or 
        pd.isnull(row.get('Age')) or row['SBP'] == 0):
        features['age_normalised_shock_index'] = np.nan
    else:
        features['age_normalised_shock_index'] = age_normalised_shock_index(row)

    # Modified Shock Index: HR/(MAP * Age)
    if (pd.isnull(row.get('HR')) or pd.isnull(row.get('MAP')) or 
        pd.isnull(row.get('Age')) or row['MAP'] == 0):
        features['modified_shock_index'] = np.nan
    else:
        features['modified_shock_index'] = modfied_shock_index(row)

    # BUN/Creatinine Ratio
    if pd.isnull(row.get('BUN')) or pd.isnull(row.get('Creatinine')) or row['Creatinine'] == 0:
        features['bun_cr'] = np.nan
    else:
        features['bun_cr'] = bun_cr(row)

    # SaO2/FiO2 Ratio
    if pd.isnull(row.get('SaO2')) or pd.isnull(row.get('FiO2')) or row['FiO2'] == 0:
        features['sao2_fio2'] = np.nan
    else:
        features['sao2_fio2'] = sao2_fio2(row)

    # Urea/Creatinine Ratio (same as BUN/Creatinine)
    if pd.isnull(row.get('BUN')) or pd.isnull(row.get('Creatinine')) or row['Creatinine'] == 0:
        features['urea_creatinine'] = np.nan
    else:
        features['urea_creatinine'] = urea_creatinine(row)

    # Pulse Pressure: SBP - DBP
    if pd.isnull(row.get('SBP')) or pd.isnull(row.get('DBP')):
        features['pulse_pressure'] = np.nan
    else:
        features['pulse_pressure'] = pulse_pressure(row)

    # Cardiac Output Estimate: (SBP - DBP) * HR
    if pd.isnull(row.get('SBP')) or pd.isnull(row.get('DBP')) or pd.isnull(row.get('HR')):
        features['cardiac_output'] = np.nan
    else:
        features['cardiac_output'] = cardiac_output(row)

    # Partial SOFA Score:
    required_sofa_keys = ['Platelets', 'Bilirubin_total', 'MAP', 'Creatinine']
    if any(pd.isnull(row.get(key)) for key in required_sofa_keys):
        features['partial_sofa'] = np.nan
    else:
        features['partial_sofa'] = partial_sofa(row)

    return features
