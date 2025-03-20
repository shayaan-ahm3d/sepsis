import pandas as pd

def partial_sofa(df: pd.DataFrame) -> pd.Series:
    """
    Partial reconstruction of the SOFA score from columns available in the 
    sepsis dataset, using pandas DataFrame.
    """
    # Initialize all scores to 0 for each row in df
    sofa = pd.Series(0.0, index=df.index)

    # --- Coagulation (Platelets) ---
    platelets = df['Platelets']
    # Platelets >= 150 => +0 (no change needed, already 0)
    sofa[(platelets >= 100) & (platelets < 150)] += 1
    sofa[(platelets >= 50) & (platelets < 100)]  += 2
    sofa[(platelets >= 20) & (platelets < 50)]   += 3
    sofa[platelets < 20]                         += 4

    # --- Liver (Bilirubin) ---
    bilirubin = df['Bilirubin_total']
    # Bilirubin < 1.2 => +0
    sofa[(bilirubin >= 1.2) & (bilirubin <= 1.9)]   += 1
    sofa[(bilirubin > 1.9) & (bilirubin <= 5.9)]    += 2
    sofa[(bilirubin > 5.9) & (bilirubin <= 11.9)]   += 3
    sofa[bilirubin > 11.9]                          += 4

    # --- Cardiovascular (MAP) ---
    map_ = df['MAP']
    # MAP >= 70 => +0
    sofa[map_ < 70] += 1

    # --- Renal (Creatinine) ---
    creatinine = df['Creatinine']
    # Creatinine < 1.2 => +0
    sofa[(creatinine >= 1.2) & (creatinine <= 1.9)]   += 1
    sofa[(creatinine > 1.9) & (creatinine <= 3.4)]    += 2
    sofa[(creatinine > 3.4) & (creatinine <= 4.9)]    += 3
    sofa[creatinine > 4.9]                            += 4

    return sofa