import shap
def shap_beeswarm(X_test, model, shap_explainer=shap.TreeExplainer):
    """
    Function to plot SHAP values using a beeswarm plot.
    
    Parameters:
    - X_test: Testing feature set
    - model: Trained Random Forest model
    
    Returns:
    - None
    """
    
    shap.initjs()
    explainer = shap_explainer(model)
    shap_values = explainer(X_test)
    shape = shap_values.shape
    if len(shape) == 3:
        shap.plots.beeswarm(shap_values[:, :, 1], max_display=shap_values.shape[1])
        shap.plots.beeswarm(shap_values[:, :, 0], max_display=shap_values.shape[1])
    else:
        shap.plots.beeswarm(shap_values[:, :], max_display=shap_values.shape[1])