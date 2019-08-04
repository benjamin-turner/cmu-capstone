# coding: utf8
from __future__ import print_function, unicode_literals
import os
import sys
import joblib
import numpy as np
import benchmarking
import delivery_prediction_predict, delivery_prediction_preprocess
import extract
import menu_options
import paths
import questionary
from questionary import Separator, Choice, prompt
from menu_options import blue
import builtins


def load_main_menu():
    menu_choice = questionary.select(
        "---Main Menu---",
        choices=['Run benchmarking',
                 'Run delivery prediction',
                 'Extract and preprocess shipment data from database',
                 'Calculate similarity matrix',
                 'Train delivery prediction model',
                 'Exit'],
        style=blue).ask()
    if menu_choice == 'Run benchmarking':
        load_benchmarking_menu()
    elif menu_choice == 'Run delivery prediction':
        load_delivery_prediction_menu()
    elif menu_choice == 'Extract and preprocess shipment data from database':
        load_extract_menu()
    elif menu_choice == 'Calculate similarity matrix':
        load_benchmarking_preprocess_menu()
    elif menu_choice == 'Train delivery prediction model':
        load_delivery_prediction_train_menu()
    elif menu_choice == 'Exit':
        sys.exit()


def load_benchmarking_menu():
    # Get matrix and KPI database
    benchmarking_matrix_kpidatabase_choice = prompt(menu_options.benchmarking_matrix_kpidatabase, style=blue)
    print("Loading saved matrix and KPI database...")
    preloaded_matrix = joblib.load(benchmarking_matrix_kpidatabase_choice['similarity_matrix'])
    builtins.sid_list = preloaded_matrix.columns.values
    preloaded_KPIs = joblib.load(benchmarking_matrix_kpidatabase_choice['kpi_database'])
    print("Matrix and KPI database loaded.")
    get_kpi(preloaded_matrix, preloaded_KPIs)
    

def get_kpi(preloaded_matrix, preloaded_KPIs):
    benchmarking_kpi_choice = prompt(menu_options.benchmarking_kpi, style=blue)
    sid = benchmarking_kpi_choice['sid']
    percentile = benchmarking_kpi_choice['percentile']
    kpi_selected = benchmarking_kpi_choice['kpi_selected']
    # Display metrics for user selection
    benchmarking.get_selected_metrics(kpi_selected, sid, percentile, preloaded_matrix, preloaded_KPIs)
    # Display menu
    choice_after = questionary.select(
        "---Run Benchmarking---",
        choices=['Run benchmarking for another SID',
                 'Go back to main menu',
                 'Exit'],
        style=blue).ask()
    if choice_after == 'Run benchmarking for another SID':
        get_kpi()
    elif choice_after == 'Go back to main menu':
        load_main_menu()
    elif choice_after == 'Exit':
        sys.exit()
        
        
def load_delivery_prediction_menu():
    # Get model and corresponding scaler and feature names
    delivery_prediction_model_choice = prompt(menu_options.delivery_prediction_model, style=blue)
    model_path = delivery_prediction_model_choice['delivery_prediction_model_choice']
    model_id = model_path.split('_', )[1].split('.', )[0]
    scaler_path = os.path.join(paths.model_scaler_dir, "scaler_") + model_id + ".pkl.z"
    feature_names_path = os.path.join(paths.data_delivery_prediction_features_dir, "feature_names_") + model_id + ".npz"
    scaler_exists = os.path.exists(scaler_path)
    feature_names_exists = os.path.exists(feature_names_path)
    print("\nScaler and feature name files are automatically selected based on model_id string after the first underscore")
    print(f"Model_id found: {model_id}")
    print(f"Looking for scaler in {scaler_path}, File exists: {scaler_exists}")
    print(f"Looking for feature names in {feature_names_path}, File exists: {feature_names_exists}")
    if scaler_exists and feature_names_exists:
        print("\nLoading model, scaler and feature names...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = np.load(feature_names_path, allow_pickle=True)
        print("Model, scaler and feature names loaded.\n")
        get_prediction(model, scaler, feature_names)
    else:
        print("Please make sure scaler/feature name files are stored in the correct directories")
        print("Returning to main menu...")
        load_main_menu()


def get_prediction(model, scaler, feature_names):
    # Display menu
    prediction_choice = questionary.select(
        "Which prediction to run?",
        choices=['Predict cost savings for one shipment',
                 'Predict delivery time window probability for batch of shipments',
                 'Go back to main menu'],
        style = blue).ask()
    if prediction_choice == 'Predict cost savings for one shipment':
        predict_one(model, scaler, feature_names)
    elif prediction_choice == 'Predict delivery time window probability for batch of shipments':
        predict_batch(model, scaler, feature_names)
    elif prediction_choice == 'Go back to main menu':
        load_main_menu()
    get_prediction()


def predict_one(model, scaler, feature_names):
    predict_one_choice = prompt(menu_options.delivery_prediction_one_questions, style=blue)
    delivery_prediction_predict.predict_one_cost_savings(predict_one_choice['shipment_date'],
                                                          predict_one_choice['shipper'],
                                                          predict_one_choice['weight'],
                                                          predict_one_choice['sender_zip'],
                                                          predict_one_choice['recipient_zip'],
                                                          model, scaler, feature_names)
    # Display menu
    choice_after = questionary.select(
        "---Predict cost savings for one shipment---",
        choices=['Predict cost savings for another shipment',
                 'Predict delivery time window probability for batch of shipments',
                 'Load another model for prediction',
                 'Go back to main menu',
                 'Exit'],
        style=blue).ask()
    if choice_after == 'Predict cost savings for another shipment':
        predict_one(model, scaler, feature_names)
    elif choice_after == 'Predict delivery time window probability for batch of shipments':
        predict_batch(model, scaler, feature_names)
    elif choice_after == 'Load another model for prediction':
        load_delivery_prediction_menu()
    elif choice_after == 'Go back to main menu':
        load_main_menu()
    elif choice_after == 'Exit':
        sys.exit()


def predict_batch(model, scaler, feature_names):
    # TODO: predict_batch
    pass


def load_extract_menu():
    extract_data_choice = prompt(menu_options.extract_data_questions, style=blue)
    records = extract.batch_query(extract_data_choice['start_date'],
                                  extract_data_choice['end_date'],
                                  extract_data_choice['sample_size'])
    extract.store(records, extract_data_choice['sample_size'])
    # Display menu
    choice_after = questionary.select(
        "---Extract and preprocess shipment data from database---",
        choices=['Extract data again',
                 'Go back to main menu',
                 'Exit'],
        style=blue).ask()
    if choice_after == 'Extract data again':
        load_extract_menu()
    elif choice_after == 'Go back to main menu':
        load_main_menu()
    elif choice_after == 'Exit':
        sys.exit()


def load_benchmarking_preprocess_menu():
    # Load extracted data
    extracted_data = load_extracted_data()
    calculate_matrix(extracted_data)


def calculate_matrix(extracted_data):
    # Get user input for metric weights
    metric_weights = {
        'weight_vs': 1 / 6,
        'weight_vpz': 1 / 6,
        'weight_vpm': 1 / 6,
        'weight_ws': 1 / 6,
        'weight_wpz': 1 / 6,
        'weight_wpm': 1 / 6
    }
    EVEN = questionary.confirm("Create a new similarity matrix with even matrix weights?").ask()
    if not EVEN:
        print("\nPlease enter weight for each matrix. Please make sure that sum of all 6 weights is approximately = 1")
        metric_weights = prompt(menu_options.benchmarking_metric_weights, style=blue)
        metric_weights_sum = sum(metric_weights.values())
        print("\nRecalibrating weights...")
        for weight in metric_weights:
            metric_weights[weight] = metric_weights[weight] / metric_weights_sum
    print("\nWeights:", metric_weights, "\n")
    # TODO: calculate matrix
    choice_after = questionary.select(
        "---Calculate similarity matrix---",
        choices=['Calculate another similarity matrix',
                 'Go back to main menu',
                 'Exit'],
        style=blue).ask()
    if choice_after == 'Calculate another similarity matrix':
        calculate_matrix(extracted_data)
    elif choice_after == 'Go back to main menu':
        load_main_menu()
    elif choice_after == 'Exit':
        sys.exit()


def load_delivery_prediction_train_menu():
    print("\nComplete steps to train a model: Extract data -> Preprocess data -> Train model")
    print("\nIf you have yet to extract a new dataset, please do so with Option 1. \nAlternatively, execute Option 3 from the Main Menu.")
    print("\nIf you have already extracted a new dataset, please choose Option 2: Preprocess data -> Train model")
    print("\nPlease only choose Option 3: Train model if you have trained a model previously, \nand only wish to try again on the same dataset with different hyperparameters")
    train_steps = questionary.select(
        "Select steps to execute",
        choices=['Extract data -> Preprocess data -> Train model',
                 'Preprocess data -> Train model',
                 'Train model'],
        style=blue).ask()
    if train_steps == 'Extract data -> Preprocess data -> Train model':
        load_extract_menu()
    elif train_steps == 'Preprocess data -> Train model':
        extracted_data = load_extracted_data()
        data_dict, model_id = preprocess(extracted_data)
        train(data_dict, model_id)
    elif train_steps == 'Train model':
        # Choose data_dict
        # Extract model_id from data_dict
        train(data_dict, model_id)
    # Display menu
    choice_after = questionary.select(
        "---Train delivery prediction model---",
        choices=['Train another model on same data but with different parameters',
                 'Go back to main menu',
                 'Exit'],
        style=blue).ask()
    if choice_after == 'Train another model on same data but with different parameters':
        train(data_dict)
    elif choice_after == 'Go back to main menu':
        load_main_menu()
    elif choice_after == 'Exit':
        sys.exit()


def preprocess(extracted_data):
    data_dict, model_id = delivery_prediction_preprocess.preprocess(extracted_data)
    return data_dict, model_id


def train(data_dict, model_id):
    # Prompt for user input for parameters e.g. estimators, depth
    n_trees = questionary.text(
        "Enter number of trees",
        style=blue,
        default=25,
        validate=menu_options.NumberValidator,
        filter=lambda val: int(val)).ask()    
    depth = questionary.text(
        "Enter maximum depth",
        style=blue,
        default=50,
        validate=menu_options.NumberValidator,
        filter=lambda val: int(val)).ask()
    # TODO: train
    pass


def load_extracted_data():
    # Display menu
    data_extracted = questionary.select(
        "You will need to extract new data before training a new model/calculating a new similarity matrix",
        choices=['I have already extracted data',
                 'I need to extract new data',
                 'Go back to main menu'],
        style=blue).ask()
    # Load model
    if data_extracted == 'I have already extracted data':
        extracted_data_choice = prompt(menu_options.extracted_data, style=blue)
        print("\nLoading data file...")
        extracted_data = joblib.load(extracted_data_choice['extracted_data_choice'])
        print("Data file loaded.\n")
        return extracted_data
    elif data_extracted == 'I need to extract new data':
        load_extract_menu()
    elif data_extracted == 'Go back to main menu':
        load_main_menu()


print("Welcome to the 71bs dashboard")
load_main_menu()
