import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

def plot_roc_auc(model, X_test, y_test):
    """
    Plots the ROC curve and prints the AUC for the subset of data where:
      - The model predicted class 1, OR
      - The true label is 1.
    
    Parameters
    ----------
    model  : trained classifier (e.g., XGBClassifier)
    X_test : test features (DataFrame or array)
    y_test : test labels (Series or array)
    """
    # Get the model's hard predictions (0 or 1)
    y_pred = model.predict(X_test)
    
    # Identify the subset of rows to keep:
    #   any row where the model predicted 1 (y_pred == 1)
    #     OR the true label is 1 (y_test == 1)
    subset_mask = (y_pred == 1) | (y_test == 1)
    
    # Subset the data
    X_subset = X_test[subset_mask]
    y_subset = y_test[subset_mask]
    
    # Predicted probabilities on the subset
    y_probs_subset = model.predict_proba(X_subset)[:, 1]
    
    # Compute ROC curve on the subset
    fpr, tpr, thresholds = roc_curve(y_subset, y_probs_subset)
    
    # Calculate AUC on the subset
    auc_val = roc_auc_score(y_subset, y_probs_subset)
    print(f"Subset ROC AUC (predicted=1 or actual=1): {auc_val:.3f}")
    
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Chance Level')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Predicted=1 or Actual=1 Only)')
    plt.legend(loc='lower right')
    plt.show()

def plot_most_important_features(model, feature_names, top_n=10):
    """
    Plots a bar chart of the top N most important features for a trained XGBoost model.
    
    Parameters
    ----------
    model : xgb.XGBClassifier or a similar scikit-learn estimator
        A trained XGBoost model with a feature_importances_ attribute.
    feature_names : list of str
        List of feature names corresponding to the columns in the training data.
    top_n : int
        Number of top features to plot (by importance).
    """
    importances = model.feature_importances_
    
    # Safety check: if the model doesn't provide importances or if
    # the lengths mismatch, handle gracefully
    if importances is None or len(importances) == 0:
        print("No feature importances found in the model.")
        return
    if len(importances) != len(feature_names):
        print("Warning: length mismatch between importances and feature_names.")
    
    importances_pct = importances / importances.sum() * 100.0
    
    indices = np.argsort(importances_pct)[::-1]  # highest first
    
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances_pct[top_indices]
    
    # Reverse again so that the top feature appears at the top of the bar chart
    top_features_reversed = top_features[::-1]
    top_importances_reversed = top_importances[::-1]
    
    plt.figure()
    plt.barh(range(top_n), top_importances_reversed)
    plt.yticks(range(top_n), top_features_reversed)
    plt.xlabel("Feature Importance (%)")
    plt.title(f"Top {top_n} Most Important Features (in %)")
    
    plt.xlim([0, max(top_importances) * 1.1])  
    
    plt.show()