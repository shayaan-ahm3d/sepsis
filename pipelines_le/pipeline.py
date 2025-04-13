import marimo

__generated_with = "0.12.8"
app = marimo.App()


@app.cell
def _():
    import lgbm_pipeline.feature_load as loader
    import lgbm_pipeline.feature_extraction as extractor

    from tqdm import tqdm
    import fireducks.pandas as pd
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext fireducks.ipyext
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import fbeta_score, make_scorer, RocCurveDisplay, ConfusionMatrixDisplay, classification_report
    import xgboost as xgb
    return (
        ConfusionMatrixDisplay,
        RocCurveDisplay,
        StratifiedKFold,
        classification_report,
        extractor,
        fbeta_score,
        loader,
        make_scorer,
        pd,
        tqdm,
        train_test_split,
        xgb,
    )


@app.cell
def _(loader):
    patients = loader.load_data("../training_set?/*.psv", max_files=None)
    return (patients,)


@app.cell
def _(patients, pd):
    patients_1 = pd.concat(patients)
    return (patients_1,)


@app.cell
def _(patients_1, pd, tqdm):
    sepsis_patients: list[pd.DataFrame] = []
    non_sepsis_patients: list[pd.DataFrame] = []
    for _patient in tqdm(patients_1, 'Splitting sepsis/non-sepsis'):
        if _patient['SepsisLabel'].any():
            sepsis_patients.append(_patient)
        else:
            non_sepsis_patients.append(_patient)
    return non_sepsis_patients, sepsis_patients


@app.cell
def _(non_sepsis_patients, sepsis_patients, train_test_split):
    train_sepsis_patients, test_sepsis_patients = train_test_split(sepsis_patients, random_state=42)
    train_non_sepsis_patients, test_non_sepsis_patients = train_test_split(non_sepsis_patients, random_state=42)
    return (
        test_non_sepsis_patients,
        test_sepsis_patients,
        train_non_sepsis_patients,
        train_sepsis_patients,
    )


@app.cell
def _(train_non_sepsis_patients, train_sepsis_patients):
    from sklearn.utils import shuffle
    train_non_sepsis_patients_1 = shuffle(train_non_sepsis_patients, random_state=42, n_samples=len(train_sepsis_patients))
    return shuffle, train_non_sepsis_patients_1


@app.cell
def _(
    pd,
    test_non_sepsis_patients,
    test_sepsis_patients,
    train_non_sepsis_patients_1,
    train_sepsis_patients,
):
    ratio: float = len(train_non_sepsis_patients_1) / len(train_sepsis_patients)
    print(f'Ratio: {ratio}')
    train_patients: list[pd.DataFrame] = train_sepsis_patients + train_non_sepsis_patients_1
    test_patients: list[pd.DataFrame] = test_sepsis_patients + test_non_sepsis_patients
    print(f'Number of sepsis patients in training set: {len(train_sepsis_patients)}')
    print(f'Number of non-sepsis patients in training set: {len(train_non_sepsis_patients_1)}')
    print(f'Number of patients in training set: {len(train_patients)}\n')
    print(f'Number of sepsis patients in testing set: {len(test_sepsis_patients)}')
    print(f'Number of non-sepsis patients in testing set: {len(test_non_sepsis_patients)}')
    print(f'Number of patients in testing set: {len(test_patients)}')
    return ratio, test_patients, train_patients


@app.cell
def _(extractor, pd, train_patients):
    train_patients_forward: list[pd.DataFrame] = extractor.fill(train_patients, extractor.FillMethod.FORWARD)
    train_patients_backward: list[pd.DataFrame] = extractor.fill(train_patients, extractor.FillMethod.BACKWARD)
    train_patients_linear: list[pd.DataFrame] = extractor.fill(train_patients, extractor.FillMethod.LINEAR)
    return (
        train_patients_backward,
        train_patients_forward,
        train_patients_linear,
    )


@app.cell
def _(
    extractor,
    pd,
    train_patients,
    train_patients_backward,
    train_patients_forward,
    train_patients_linear,
):
    fill_method_to_train_patients: dict[extractor.FillMethod, list[pd.DataFrame]] = {
    	extractor.FillMethod.FORWARD : train_patients_forward,
    	extractor.FillMethod.BACKWARD: train_patients_backward,
    	extractor.FillMethod.LINEAR  : train_patients_linear}
    fill_methods_to_use: dict[str, extractor.FillMethod] = extractor.best_fill_method_for_feature(
    	fill_method_to_train_patients)
    train_patients_mixed: list[pd.DataFrame] = extractor.mixed_fill(train_patients, train_patients_forward,
                                                                    train_patients_backward, train_patients_linear,
                                                                    fill_methods_to_use)
    return (
        fill_method_to_train_patients,
        fill_methods_to_use,
        train_patients_mixed,
    )


@app.cell
def _(extractor, fill_methods_to_use, pd, test_patients):
    test_patients_forward: list[pd.DataFrame] = extractor.fill(test_patients, extractor.FillMethod.FORWARD)
    test_patients_backward: list[pd.DataFrame] = extractor.fill(test_patients, extractor.FillMethod.BACKWARD)
    test_patients_linear: list[pd.DataFrame] = extractor.fill(test_patients, extractor.FillMethod.LINEAR)
    test_patients_mixed: list[pd.DataFrame] = extractor.mixed_fill(test_patients, test_patients_forward,
                                                                   test_patients_backward, test_patients_linear,
                                                                   fill_methods_to_use)
    return (
        test_patients_backward,
        test_patients_forward,
        test_patients_linear,
        test_patients_mixed,
    )


@app.cell
def _(create_windows, test_patients_mixed, train_patients_mixed):
    for _patient in train_patients_mixed:
        create_windows(_patient)
    for _patient in test_patients_mixed:
        create_windows(_patient)
    return


@app.cell
def _(pd, test_patients_mixed, train_patients_mixed):
    train = pd.concat(train_patients_mixed)
    test = pd.concat(test_patients_mixed)

    X_train = train.drop(columns="SepsisLabel", inplace=False)
    y_train = train["SepsisLabel"]
    X_test = test.drop(columns="SepsisLabel", inplace=False)
    y_test = test["SepsisLabel"]
    return X_test, X_train, test, train, y_test, y_train


@app.cell
def _(X_train, fbeta_score, make_scorer, ratio, xgb, y_train):
    f = make_scorer(fbeta_score, beta=5.5)

    clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", scale_pos_weight=ratio)
    bst = clf.fit(X_train, y_train)
    return bst, clf, f


@app.cell
def _(X_test, bst):
    y_pred = bst.predict(X_test)
    return (y_pred,)


@app.cell
def _(
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    y_pred,
    y_test,
):
    RocCurveDisplay.from_predictions(y_test, y_pred)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    return


if __name__ == "__main__":
    app.run()
