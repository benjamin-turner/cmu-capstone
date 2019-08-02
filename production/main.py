# coding: utf8

from __future__ import print_function, unicode_literals
import os
import sys
import joblib
import numpy as np
from PyInquirer import prompt
from examples import custom_style_3
import benchmarking
import delivery_prediction_predict
import extract
import menu_options
import paths


def load_main_menu():
    main_menu_choice = prompt(menu_options.main_menu, style=custom_style_3)
    user_choice = menu_options.main_menu_options.index(main_menu_choice['main_menu']) + 1
    if user_choice == 1:
        load_benchmarking_menu()
    if user_choice == 2:
        load_delivery_prediction_menu()
    if user_choice == 3:
        load_extract_menu()
    if user_choice == 4:
        load_benchmarking_preprocess_menu()
    if user_choice == 5:
        # TODO: Royce code for preprocessing and training
        sys.exit()
    else:
        sys.exit()


def load_benchmarking_menu():
    # Get user input for SID and pecentile
    benchmarking_initial_answers = prompt(menu_options.benchmarking_initial_questions, style=custom_style_3)
    sid = benchmarking_initial_answers['sid']
    percentile = benchmarking_initial_answers['percentile']
    benchmarking_metric_selection_answers = prompt(menu_options.benchmarking_metrics_selection_questions, style=custom_style_3)
    metrics_selected = benchmarking_metric_selection_answers['benchmarking_metrics_selection']
    # Display metrics for user selection
    benchmarking.get_selected_metrics(metrics_selected, sid, percentile, preloaded_matrix, preloaded_KPIs)
    # Display menu
    benchmarking_menu_choice = prompt(menu_options.benchmarking_menu, style=custom_style_3)
    user_choice = menu_options.benchmarking_menu_options.index(benchmarking_menu_choice['benchmarking_menu']) + 1
    if user_choice == 1:
        load_benchmarking_menu()
    elif user_choice == 2:
        load_main_menu()
    else:
        sys.exit()


def load_delivery_prediction_menu():
    delivery_prediction_menu_choice = prompt(menu_options.delivery_prediction_menu, style=custom_style_3)
    user_choice = menu_options.delivery_prediction_menu_options.index(delivery_prediction_menu_choice['delivery_prediction_menu']) + 1
    # Get user choice for model and scaler
    model_menu_choice = prompt(menu_options.model_menu, style=custom_style_3)
    model_path = model_menu_choice['model_menu']
    scaler_dir_scaler_ = os.path.join(paths.model_scaler_dir,"scaler_")
    scaler_path = scaler_dir_scaler_ + model_path.split('_', )[1]
    if user_choice == 1:
        load_predict_one_menu(scaler_path, model_path)
    elif user_choice == 2:
        # TODO: load_predict_batch_menu
        sys.exit()
    else:
        sys.exit()


def load_predict_one_menu(scaler_path, model_path):
    load_predict_one_questions(scaler_path, model_path)
    while True:
        predict_one_menu_choice = prompt(menu_options.predict_one_menu, style=custom_style_3)
        predict_one_choice = menu_options.predict_one_menu_options.index(
            predict_one_menu_choice['predict_one_menu']) + 1
        if predict_one_choice == 1:
            load_predict_one_questions(scaler_path, model_path)
        elif predict_one_choice == 2:
            load_main_menu()
        else:
            sys.exit()


def load_predict_one_questions(scaler_path, model_path):
    # Get user input for shipment_date, shipper, weight, sender_zip, recipient_zip,
    predict_one_input = prompt(menu_options.delivery_prediction_one_questions, style=custom_style_3)
    # pprint(predict_one_input)
    print(" ")
    # Run prediction
    df = delivery_prediction_predict.predict_one_cost_savings(predict_one_input['shipment_date'],
                                                              predict_one_input['shipper'],
                                                              predict_one_input['weight'],
                                                              predict_one_input['sender_zip'],
                                                              predict_one_input['recipient_zip'],
                                                              # For demo
                                                              scaler_path, preloaded_model)
                                                              # scaler_path, model_path)
    return df


def load_extract_menu():
    extract_data_answers = prompt(menu_options.extract_data_questions, style=custom_style_3)
    records = extract.batch_query(extract_data_answers['start_date'],
                                  extract_data_answers['end_date'],
                                  extract_data_answers['sample_size'])
    output_path = extract.store(records, extract_data_answers['sample_size'])
    print(f"Data extracted and stored in {output_path}")
    extract_menu_choice = prompt(menu_options.extract_menu, style=custom_style_3)
    extract_menu_choice_no = menu_options.extract_menu_options.index(
        extract_menu_choice['extract_menu']) + 1
    if extract_menu_choice_no == 1:
        load_extract_menu()
    elif extract_menu_choice_no == 2:
        load_main_menu()
    else:
        sys.exit()


def load_benchmarking_preprocess_menu():
    # Get user input for SID and pecentile
    benchmarking_preprocess_answers = prompt(menu_options.benchmarking_preprocess_questions, style=custom_style_3)
    similarity_matrix_choice = benchmarking_preprocess_answers['benchmarking_preprocess_answer']
    print(f"You have chosen to use the {similarity_matrix_choice} to generate the new similarity matrix")
    # Display menu
    benchmarking_preprocess_menu_choice = prompt(menu_options.benchmarking_preprocess_menu, style=custom_style_3)
    user_choice = menu_options.benchmarking_preprocess_menu_options.index(benchmarking_preprocess_menu_choice['benchmarking_preprocess_menu']) + 1
    if user_choice == 1:
        load_benchmarking_preprocess_menu()
    elif user_choice == 2:
        load_main_menu()
    else:
        sys.exit()

# For demo purposes, we preload the models
print("Preloading delivery prediction model...")
# preloaded_model = joblib.load(paths.model_cmu)
preloaded_model = "FILLER"
print("Model preloaded.")

print("Preloading benchmarking data...")
# Preload the similarity score matrix
preloaded_matrix = joblib.load(paths.benchmarking_ss_matrix_cmu)
# Convert SS matrix to np array for performance
full_ss_matrix_optimized = np.array(preloaded_matrix)
# Collect all sid values from SS matrix
full_matrix_indices = np.array(preloaded_matrix.index.values)
import builtins
builtins.sid_list = preloaded_matrix.columns.values
# Prelaod the KPI database
preloaded_KPIs = joblib.load(paths.benchmarking_kpi_cmu)
# Convert KPI database to np array for performance
full_kpi_optimized = np.array(preloaded_KPIs)
# Collect all sid values from KPI database
full_kpi_indices = np.array(preloaded_KPIs.index.values)
print("Benchmarking data preloaded.")

print("Welcome to the 71bs dashboard")
load_main_menu()





