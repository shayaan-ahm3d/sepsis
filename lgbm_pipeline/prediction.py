import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import xgboost as xgb
import feature_load as loader
import feature_extraction as cleaner
import feature_plots as plotter

# sliding window to create feature vector of means min max of continuous features
# Upsample data imbalance sepsis / not-sepsis

df_raw = loader.loadTrainingData(path_pattern='../training_setA/*.psv', max_files=100000)
df_clean = cleaner.cleanData(df_raw)

def create_window_feature_vector(patient_df, window=6):
  X = patient_df.drop(columns=["SepsisLabel"])
  y = patient_df["SepsisLabel"]

  df_mean = X.rolling(window).mean().add_suffix("_mean")
  df_min  = X.rolling(window).min().add_suffix("_min")
  df_max  = X.rolling(window).max().add_suffix("_max")
  df_std  = X.rolling(window).std().add_suffix("_std")

  X_rolled = pd.concat([df_mean, df_min, df_max, df_std], axis=1)
  X_rolled = X_rolled.iloc[window-1:].reset_index(drop=True)
  y_aligned = y.iloc[window-1:].reset_index(drop=True)
    
  # Return the rolling features + label for this patient
  return X_rolled, y_aligned


X_all,y_all = create_window_feature_vector(df_clean, window=6)

# for subdf in df_clean:
#     print("Type of subdf:", type(subdf))
#     Xp, yp = create_window_feature_vector(subdf, window=6)
#     dfs.append(Xp)
#     ys.append(yp)

print("Shape of X_rolled:", X_all.shape)
print("Shape of y_aligned:", y_all.shape)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    shuffle=True)

model = xgb.XGBClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

plotter.plot_roc_auc(model, X_test, y_test)