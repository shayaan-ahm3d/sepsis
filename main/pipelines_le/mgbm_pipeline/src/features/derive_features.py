import numpy as np
import pandas as pd
import polars as pl

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

def compute_derived_features_polars(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        # Shock Index
        (pl.when(pl.col("SBP") > 0)
         .then(pl.col("HR") / pl.col("SBP"))
         .otherwise(None))
         .cast(pl.Float32)
         .alias("ShockIndex"),
        
        # Age Normalised Shock Index
        (pl.when((pl.col("SBP") > 0) & (pl.col("Age") > 0))
         .then(pl.col("HR") / (pl.col("SBP") * pl.col("Age")))
         .otherwise(None))
         .cast(pl.Float32)
         .alias("AgeNormalisedShockIndex"),
        
        # Modified Shock Index
        (pl.when((pl.col("MAP") > 0) & (pl.col("Age") > 0))
         .then(pl.col("HR") / (pl.col("MAP") * pl.col("Age")))
         .otherwise(None)).alias("ModifiedShockIndex"),
        
        # BUN/Creatinine Ratio
        (pl.when(pl.col("Creatinine") > 0)
         .then(pl.col("BUN") / pl.col("Creatinine"))
         .otherwise(None))
         .cast(pl.Float32)
         .alias("UCR"),
        
        # SaO2/FiO2 Ratio 
        (pl.when(pl.col("FiO2") > 0)
         .then(pl.col("SaO2") / pl.col("FiO2"))
         .otherwise(None))
         .cast(pl.Float32)
         .alias("SaO2_FiO2"),
        
        # Pulse Pressure
        (pl.col("SBP") - pl.col("DBP"))
        .cast(pl.Float32)
        .alias("PulsePressure"),
    ])
    
    # Add cardiac output (requires pulse_pressure)
    df = df.with_columns(
        (pl.col("PulsePressure") * pl.col("HR"))
        .cast(pl.Float32)
        .alias("CardiacOutput")
    )
    
    # Calculate SOFA
    sofa = (
        # Start with 0
        pl.lit(0) + 
        # Platelets component
        pl.when((df["Platelets"] >= 100) & (df["Platelets"] < 150)).then(1)
        .when((df["Platelets"] >= 50) & (df["Platelets"] < 100)).then(2)
        .when((df["Platelets"] >= 20) & (df["Platelets"] < 50)).then(3)
        .when(df["Platelets"] < 20).then(4)
        .otherwise(0) +
        # Bilirubin component
        pl.when((df["Bilirubin_total"] >= 1.2) & (df["Bilirubin_total"] <= 1.9)).then(1)
        .when((df["Bilirubin_total"] > 1.9) & (df["Bilirubin_total"] <= 5.9)).then(2)
        .when((df["Bilirubin_total"] > 5.9) & (df["Bilirubin_total"] <= 11.9)).then(3)
        .when(df["Bilirubin_total"] > 11.9).then(4)
        .otherwise(0) +
        # MAP component
        pl.when(df["MAP"] < 70).then(1).otherwise(0) +
        # Creatinine component
        pl.when((df["Creatinine"] >= 1.2) & (df["Creatinine"] <= 1.9)).then(1)
        .when((df["Creatinine"] > 1.9) & (df["Creatinine"] <= 3.4)).then(2)
        .when((df["Creatinine"] > 3.4) & (df["Creatinine"] <= 4.9)).then(3)
        .when(df["Creatinine"] > 4.9).then(4)
        .otherwise(0)
    ).cast(pl.Float32).alias("PartialSOFA")
    
    # Calculate qSOFA (Quick SOFA)
    qsofa = (
        # Start with 0
        pl.lit(0) +
        # SBP component
        pl.when(df["SBP"] <= 100).then(1).otherwise(0) +
        # Respiratory rate component
        pl.when(df["Resp"] >= 22).then(1).otherwise(0)
    ).cast(pl.Float32).alias("qSOFA")

    # Add SOFA score
    df = df.with_columns([sofa, qsofa])
    df = df.drop(["HR", "SBP","DBP", "MAP", "Creatinine", "BUN", "Bilirubin_total", "Platelets", "Resp", "FiO2", "SaO2"])
    
    return df
