import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_missingness(df: pd.DataFrame, title: str = "Missing Data Visualization") -> None:
    """
    Graphs the percentage of missing data for each column in the DataFrame.

    Parameters:
      df: pd.DataFrame - the DataFrame to analyze.
      title: str - the title for the plot.
    """
    # Calculate percentage of missing values per column and filter out columns with no missing values.
    missing_percent = df.isnull().mean() * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(missing_percent.index, missing_percent.values, color="skyblue")
    plt.xlabel("Percentage Missing")
    plt.title(title)
    
    # Annotate each bar with the missing percentage value.
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

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