"""
Delivery Prediction Training Script

This script contains the functions to train the model and print relevant training results for the user in
excel format.

Model and model stats are saved as acc-<accuracy>-model_<model_id>_.pkl.z and acc-<accuracy>-model_<model_id>_.xlsx
with model_id being the YYYYMMDD-HHMM timestamp created when preprocessing was initialized
and accuracy being the model validation accuracy.

This file should be imported as a module and contains the following
function that is used in main.py:
    * train - trains a random forest model on preprocessed data.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import paths
import utilities


def get_accuracy_windows(max_windows, y_test, y_pred):
    """Calculates and stores model validation accuracy as dataframe.

    Args:
        max_windows: Maximum +- windows.
        y_test: Ground truth labels of target variable.
        y_pred: Predicted labels of target variable.

    Returns:
         pandas dataframe obj: Pandas dataframe with accuracy for +- max_windows

    """
    # Initialize array to hold counts for each window
    count_arr = np.zeros(max_windows)

    # For each class window, if predicted class is in window, increment count
    # E.g. if predicted class = 4 and target class = 6, since max window allowed = 2, consider instance as accurate and increment count
    for idx, value in enumerate(y_test):
        for window in np.arange(1, max_windows + 1):
            # window_arr calculates window that predicted value can fall into
            # e.g. target value = 4, window = 2, window_arr = {2,3,4,5,6}
            window_arr = np.arange(value - window, value + window + 1)
            if (y_pred[idx] in window_arr):
                count_arr[window - 1] += 1

    # Print accuracy for each time window
    accuracy_dict = {}
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Accuracy with +- 0 time window(s): {accuracy:.4f}%")
    accuracy_dict[0] = f"{accuracy:.4f}%"
    for idx, count in enumerate(count_arr):
        accuracy = (count / len(y_pred)) * 100
        print(f"Accuracy with +- {idx + 1} time window(s): {accuracy:.4f}%")
        accuracy_dict[idx+1] = f"{accuracy:.4f}%"
    accuracy_df = pd.DataFrame([accuracy_dict]).T
    accuracy_df.columns = ['Accuracy']
    accuracy_df.index.rename('+- Time Windows', inplace=True)
    return accuracy_df


def get_feature_importance(model, model_id):
    """Obtains and stores feature importance in % as dataframe.

    Args:
        model (obj): Model used for training and predicting dataset.
        model_id (str): Timestamp used to identify model, scaler and feature names files

    Returns:
        pandas dataframe obj: Pandas dataframe with importance of each feature used.
    """
    df = pd.DataFrame()
    # Load feature names
    features_path = os.path.join(paths.data_delivery_prediction_features_dir, "feature_names_" + model_id + ".npz")
    feature_names = np.load(features_path, allow_pickle=True)
    df['features'] = feature_names['feature_names_dummified']
    # Get feature importance
    df['importance'] = model.feature_importances_
    df.sort_values(by=['importance'], ascending=False, inplace=True)
    return df


def get_params(model):
    """Get model parameters and store in dataframe.

    Args:
        model (obj): Model used for training and predicting dataset.

    Returns:
        pandas dataframe obj: Pandas dataframe with model parameters
    """
    params_dict = model.get_params()
    return pd.DataFrame([params_dict]).T


def get_classification_report(y_test, y_pred):
    """Get report on precision and f1-score for each class and store in dataframe.

    Args:
        y_test: Ground truth labels of target variable.
        y_pred: Predicted labels of target variable.

    Returns:
        pandas dataframe obj: Pandas dataframe with classification report
    """
    windows_cmu = joblib.load(paths.windows_cmu)
    report_dict = classification_report(y_test, y_pred, output_dict=True, target_names=list(windows_cmu.values()))
    classification_report_df = pd.DataFrame([report_dict]).T
    classification_report_df.columns = ['Importance']
    classification_report_df = classification_report_df['Importance'].apply(pd.Series)
    classification_report_df.index.rename('Time Window', inplace=True)
    return classification_report_df


def train(datadict, model_id, n_estimators=25, max_depth=50):
    """Random forest model training.

    Trains random forest model. Only n_estimators, max_depth hyperparameters are available to the user for training.
    The rest of the hyperparameters have been tuned by the CMU team.
    Saves model and statistics after training.

    Args:
        datadict (dict): Dictionary of numpy arrays containing preprocessed train and test data.
        model_id (str): Timestamp used to identify model, scaler and feature names files.
        n_estimators (str): Number of trees in forest. Less likely to overfit with more trees.
        max_depth (str): The maximum depth of the tree. More likely to overfit if depth is large.

    """
    # Convert n_estimators and max_depth from string to int since model only accepts int
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    print("\nTraining model...")
    start_time = time.time()
    X_train = datadict['X_train']
    y_train = datadict['y_train']
    X_test = datadict['X_test']
    y_test = datadict['y_test']
    model_rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, verbose=1, n_jobs=-1, bootstrap=False)
    print("\nFitting model...")
    print("Parameters used:", model_rf.get_params())
    model_rf.fit(X_train, y_train)
    print("\nPredicting results...")
    y_pred_rf = model_rf.predict(X_test)
    # y_pred_proba_rf = model_rf.predict_proba(X_test)
    print("\nCalculating accuracy...")
    accuracy_df = get_accuracy_windows(1, y_test, y_pred_rf)
    accuracy = accuracy_score(y_test, y_pred_rf) * 100
    # Save model
    print("\nSaving model...")
    model_path = os.path.join(paths.model_dir, "acc-" + f"{accuracy:.2f}" + "-model_" + model_id + ".pkl.z")
    print(f"Model saved in {model_path}")
    joblib.dump(model_rf, model_path)
    # Get model stats
    feature_importance_df = get_feature_importance(model_rf, model_id)
    classification_report_df = get_classification_report(y_test, y_pred_rf)
    params_df = get_params(model_rf)
    # Save stats to excel
    print("\nSaving model stats...")
    stats_path = os.path.join(paths.output_delivery_prediction_stats_dir, "acc-" + f"{accuracy:.2f}" + "-stats_" + model_id + ".xlsx")
    print(f"Stats saved in {stats_path}")
    with pd.ExcelWriter(stats_path) as writer:
        accuracy_df.to_excel(writer, sheet_name='Accuracy')
        feature_importance_df.to_excel(writer, sheet_name='Feature Importance')
        classification_report_df.to_excel(writer, sheet_name='Classification Report')
        params_df.to_excel(writer, sheet_name='Model Parameters')

    utilities.print_elapsed_time(start_time)

if __name__ == '__main__':
    datadict_file = "datadict_sample_cmu.pkl.z"
    datadict_path = os.path.join(paths.data_delivery_prediction_data_dict_dir, datadict_file)
    print(datadict_path)
    datadict = joblib.load(datadict_path)
    # Get model id from path
    model_id = datadict_path.split('_', )[-1].split('.', )[0]
    print("model_id:", model_id)
    train(datadict, model_id, 10, 10)
