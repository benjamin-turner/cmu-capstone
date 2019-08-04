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
    print(f"\nAccuracy with +- 0 time window(s): {accuracy:.4f}%")
    accuracy_dict[0] = f"{accuracy:.4f}%"
    for idx, count in enumerate(count_arr):
        accuracy = (count / len(y_pred)) * 100
        print(f"Accuracy with +- {idx + 1} time window(s): {accuracy:.4f}%\n")
        accuracy_dict[idx+1] = f"{accuracy:.4f}%"

    return pd.DataFrame([accuracy_dict], columns=['time windows +-', 'accuracy'])


def get_feature_importance(model, model_id):
    df = pd.DataFrame()
    # Load feature names
    features_path = os.path.join(paths.data_delivery_prediction_features_dir, "feature_names_" + model_id + ".npz")
    feature_names = np.load(features_path)
    df['features'] = feature_names['feature_names_dummified']
    # Get feature importance
    df['importance'] = model.feature_importances_
    return df


def get_params(model):
    params_dict = model.get_params()
    return pd.DataFrame([params_dict])


def get_classification_report(y_test, y_pred):
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    return pd.DataFrame([report_dict])


def train(datadict, model_id, n_estimators=25, max_depth=50):
    print("Fitting model...")
    start_time = time.time()
    X_train = datadict['X_train']
    y_train = datadict['y_train']
    X_test = datadict['X_test']
    y_test = datadict['y_test']
    model_rf = RandomForestClassifier(n_estimators, max_depth, verbose=1, n_jobs=-1, bootstrap=False)
    print("Parameters:", model_rf.get_params())
    model_rf.fit(X_train, y_train)
    print("Predicting results...")
    y_pred_rf = model_rf.predict(X_test)
    y_pred_proba_rf = model_rf.predict_proba(X_test)
    print("Calculating accuracy...")
    accuracy = accuracy_score(y_test, y_pred_rf) * 100
    # Save model
    model_path = os.path.join(paths.model_dir, "acc-" + str(accuracy) + "-model_" + model_id + "pkl.z")
    joblib.dump(model_rf, model_path)
    # Get model stats
    accuracy_df = get_accuracy_windows(1, y_test, y_pred_rf)
    feature_importance_df = get_feature_importance(model_rf, model_id)
    classification_report_df = get_classification_report(y_test, y_pred_rf)
    params_df = get_params(model_rf)
    print(accuracy_df)
    print(feature_importance_df)
    print(classification_report_df)
    print(params_df)
    # Save stats to excel
    stats_path = os.path.join(paths.output_delivery_prediction_dir, "stats_" + model_id + ".xlsx")
    with pd.ExcelWriter(stats_path) as writer:
        accuracy_df.to_excel(writer, sheet_name='Accuracy')
        feature_importance_df.to_excel(writer, sheet_name='Feature Importance')
        classification_report_df.to_excel(writer, sheet_name='Classification Report')
        params_df.to_excel(writer, sheet_name='Model Parameters')

    utilities.print_elapsed_time(start_time)

if __name__ == '__main__':
    datadict_file = "datadict_20190803-2225.pkl.z"
    datadict_path = os.path.join(paths.data_delivery_prediction_data_dict_dir, datadict_file)
    print(datadict_path)
    datadict = joblib.load(datadict_path)
    # Get model id from path
    model_id = datadict_path.split('_', )[-1].split('.', )[0]
    print("model_id:", model_id)
    # train(datadict, model_id, 10, 10)
