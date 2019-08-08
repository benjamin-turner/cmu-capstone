"""
Delivery Prediction - Model - xgBoost

This script covers the following:
- Loading of preprocessed data for training the model
- Preprocessing data for xgboost
- Training a gradient boosting model with xgBoost
- Predicting accuracy of xgBoost model

"""

import time
import os
from datetime import timedelta
from datetime import datetime
import joblib
import argparse

import numpy as np; print('numpy Version:', np.__version__)
import pandas as pd; print('pandas Version:', pd.__version__)
import xgboost as xgb; print('xgb Version:', xgb.__version__)
from sklearn.metrics import accuracy_score

def calc_accuracy_windows(max_windows, y_test, y_pred):
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
    accuracy_list = []
    print(f"Accuracy with +- 0 time window(s): {accuracy_score(y_test, y_pred) * 100:.4f}%")
    accuracy_list.append(accuracy_score(y_test, y_pred))
    for idx, count in enumerate(count_arr):
        print(f"Accuracy with +- {idx + 1} time window(s): {(count / len(y_pred)) * 100:.4f}%")
        accuracy_list.append(count / len(y_pred))

    return accuracy_list

def get_predictions(bst, X_test, y_test):
    start_time = time.time()

    # Predict and print predictions
    print("Predicting...")
    y_pred = bst.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Window Accuracy: ")
    calc_accuracy_windows(2, y_test, y_pred)

    elapsed_time_secs = time.time() - start_time
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
    print(msg)

# Arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--cont', action='store_true',
                    help="Continue training")
parser.add_argument('--model',
                    help="Model ID")
parser.add_argument('--rounds', type=int, default= 5,
                    help="Number of rounds")
args = parser.parse_args()

# Model
model_dir = 'model'
model_id = datetime.now().strftime('%Y-%m-%d--%H-%M')
all_start_time = time.time()

# Load data
print("Loading data...")
start_time = time.time()

data_dict = np.load("data/data_dict_Windows21_SMOTEno_MSAno.npz")
X_train = data_dict['X_train']
y_train = data_dict['y_train']
X_test = data_dict['X_test']
y_test = data_dict['y_test']

elapsed_time_secs = time.time() - start_time
msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
print(msg)  

if args.cont:
    # Load model for training
    print("Loading model for training")
    start_time = time.time()
    model_path = os.path.join(model_dir, args.model)
    print("Model is stored in ", model_path)
    bst = joblib.load(model_path)

    elapsed_time_secs = time.time() - start_time
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
    print(msg)

    # Get booster object from classifier
    print("Getting booster object...")
    bst_booster = bst.get_booster()

    # Update params if needed
    print("Setting new params: n_estimators = ", args.rounds)
    bst.set_params(n_estimators=args.rounds)

    # Print params
    print("Getting new params...")
    print(bst.get_xgb_params())

    # Continue training
    print("Start training...")
    start_time = time.time()
    bst.fit(X_train,y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric='mlogloss',
            early_stopping_rounds = 5,
            verbose=True,
            xgb_model = bst_booster)

    elapsed_time_secs = time.time() - start_time
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
    print(msg)

    # Get predictions
    get_predictions(bst, X_test, y_test)
    
    # Save model
    print("Saving model...")
    start_time = time.time()
    
    ## https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    joblib.dump(bst, 'model/xgbmodel-' + model_id + '.z', protocol = 4)

    elapsed_time_secs = time.time() - start_time
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
    print(msg)
    
else:
    print("No model detected, starting new model")
    # Train model
    print("Training model...")
    start_time = time.time()

    bst = xgb.XGBClassifier(max_depth=61,
                            n_estimators= args.rounds,
                            colsample_bytree=1.0,
                            learning_rate=0.1,
                            random_state= 71,
                            verbosity = 3,
                            silent = 0,
                            n_jobs=-1,
                            tree_method='gpu_hist', n_gpus = 2,
                            predictor ='cpu_predictor')

    # Print params
    print(bst.get_xgb_params())

    bst.fit(X_train,y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds = 5,
            verbose=True
           )

    print(bst.evals_result())

    elapsed_time_secs = time.time() - start_time
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
    print(msg)

    # Predict and print predictions
    print("Predicting")
    start_time = time.time()

    get_predictions(bst, X_test, y_test)

    elapsed_time_secs = time.time() - start_time
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
    print(msg)

    # Save model
    print("Saving model")
    start_time = time.time()
    
    ## https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    joblib.dump(bst, 'model/xgbmodel-' + model_id + '.z', protocol = 4)

    elapsed_time_secs = time.time() - start_time
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
    print(msg)

total_time_secs = time.time() - all_start_time
msg = "Total Execution time: %s secs (Wall clock time)" % timedelta(seconds=round(total_time_secs))
print(msg)

# # Load model for prediction
# print("Loading model for prediction")
# model_id_to_load = 1 # fill in
# bst = joblib.load('model/xgbmodel-' + model_id + '.pkl.z')
# print("Predicting...")
# get_predictions(bst, X_test, y_test)

# # Load model for training
# print("Loading model for training")
# model_id_to_load = 1 # fill in
# bst = joblib.load('model/xgbmodel-' + model_id + '.pkl.z')
# # Get booster object from classifier
# bst_booster = bst.get_booster()
# # Update params if needed
# bst.set_params(n_estimators=2)
# # Continue training
# bst.fit(X_train,y_train,
#         eval_set=[(X_train, y_train), (X_test, y_test)],
#         eval_metric='mlogloss',
#         early_stopping_rounds = 5,
#         verbose=True,
#         xgb_model = bst_booster)
# # Get predictions
# get_predictions(bst, X_test, y_test)
# # Save model
# joblib.dump(bst, 'model/xgbmodel-' + model_id + '.pkl.z')



