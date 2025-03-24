import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

def plot_missingness(df: pd.DataFrame, title: str = "Missing Data Visualization") -> None:
    """
    Graphs the percentage of missing data for each column in the DataFrame.

    Parameters:
      df: pd.DataFrame - the DataFrame to analyze.
      title: str - the title for the plot.
    """
    # Calculate percentage of missing values per column.
    missing_percent = df.isnull().mean() * 100
    
    # Print the average missingness across all columns.
    avg_missing = missing_percent.mean()
    print(f"Average missingness: {avg_missing:.2f}%")
    
    # Filter to only columns with missing values and sort them.
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
    
    # Create a vertical bar chart with wider bars.
    plt.figure(figsize=(12, 6))
    bars = plt.bar(missing_percent.index, missing_percent.values, color="skyblue", width=.7)
    plt.ylabel("Percentage Missing")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    
    # Annotate each bar with its missing percentage.
    for bar in bars:
      height = bar.get_height()
      plt.text(
          bar.get_x() + bar.get_width() / 2,
          height + 1,
          f"{height:.1f}%",
          ha="center",
          va="bottom",
          fontsize=10,
          rotation=45  # Rotate the annotation text by 45 degrees.
      )
    
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
    
def plot_confusion_matrix(y_true, y_pred, labels=("No Sepsis", "Sepsis")):
    """
    Plots a confusion matrix using matplotlib.
    
    Args:
        y_true: 1D array-like of true labels (0 or 1).
        y_pred: 1D array-like of predicted labels (0 or 1).
        labels: Tuple or list of label names for display 
                (first is the "negative" label, second is the "positive" label).
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Show the matrix with imshow. By default it uses a colormap, 
    # but we are not specifying any color explicitly.
    im = ax.imshow(cm, interpolation='nearest')
    
    # Set the tick labels to your chosen names
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Loop over data dimensions and create text annotations.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center"
            )
    
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.show()