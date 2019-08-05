# coding: utf8
"""
Dashboard UI

This script controls the UI of the app. It prompts for user input before loading files
or redirecting users to other modules.

This file should be run as the entry point into the dashboard.

"""
from __future__ import print_function, unicode_literals
import os
import sys
import joblib
import numpy as np
import questionary
from questionary import Separator, Choice, prompt
import builtins
import glob
import pandas as pd
import menu_options, paths
from menu_options import blue
import delivery_prediction_predict, delivery_prediction_preprocess, delivery_prediction_train
import benchmarking, benchmarking_calc
import extract


def load_main_menu():
    """Loads main menu of 71lbs dashboard. Invokes other functions according to user choice.

    """
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
    """Loads user prompts for running benchmarking.

    Loads matrix and KPI database according to user choice.
    Invokes get_kpi to calculate KPIs
    """
    # Get matrix and KPI database
    similarity_matrices = []
    for idx, file_path in enumerate(glob.glob(paths.data_benchmarking_dir + "/*similarity*")):
        similarity_matrices.append(file_path)
    KPI_databases = []
    for idx, file_path in enumerate(glob.glob(paths.data_benchmarking_dir + "/*KPI*")):
        KPI_databases.append(file_path)
    # Ask for user choice
    matrix_path = questionary.select(
        "Which similarity matrix to use?",
        choices=similarity_matrices,
        style=blue).ask()
    KPI_database = questionary.select(
        "Which KPI database to use?",
        choices=KPI_databases,
        style=blue).ask()
    print("Loading saved matrix and KPI database...")
    preloaded_matrix = joblib.load(matrix_path)
    builtins.sid_list = preloaded_matrix.columns.values
    preloaded_KPIs = joblib.load(KPI_database)
    print("Matrix and KPI database loaded.")
    get_kpi(preloaded_matrix, preloaded_KPIs)


def get_kpi(preloaded_matrix, preloaded_KPIs):
    """Prompts for user selection on KPIs to print.

    Invokes benchmarking.get_selected_metrics() to calculate KPIs.
    Lastly, display menu to user after calculation.

    Args:
        preloaded_matrix (obj): Matrix used for calculating KPIs
        preloaded_KPIs (obj): Metrics used for calculating KPIs

    """
    benchmarking_kpi_choice = prompt(menu_options.benchmarking_kpi, style=blue)
    sid = benchmarking_kpi_choice['sid']
    percentile = benchmarking_kpi_choice['percentile']
    kpi_selected = benchmarking_kpi_choice['kpi_selected']
    # Display metrics for user selection
    benchmarking.get_selected_metrics(kpi_selected, sid, percentile, preloaded_matrix, preloaded_KPIs)
    # Display menu
    while True:
        choice_after = questionary.select(
            "---Run Benchmarking---",
            choices=['Run benchmarking for another SID',
                     'Go back to main menu',
                     'Exit'],
            style=blue).ask()
        if choice_after == 'Run benchmarking for another SID':
            get_kpi(preloaded_matrix, preloaded_KPIs)
        elif choice_after == 'Go back to main menu':
            load_main_menu()
        elif choice_after == 'Exit':
            sys.exit()


def load_delivery_prediction_menu():
    """Get user choice for model to use for prediction and loads model, respective scaler, feature names.

    Invokes get_prediction() to after loading.
    If scaler and feature names cannot be found in their directories, redirect user to main menu.
    """
    # Get model and corresponding scaler and feature names
    models = []
    for idx, model in enumerate(glob.glob(paths.model_dir + "/*.pkl.z")):
        models.append(model)
    model_path = questionary.select(
        "Which model to use?",
        choices=models,
        style=blue).ask()
    model_id = model_path.split('_', )[-1].split('.', )[0]
    scaler_path = os.path.join(paths.model_scaler_dir, "scaler_") + model_id + ".pkl.z"
    feature_names_path = os.path.join(paths.data_delivery_prediction_features_dir, "feature_names_") + model_id + ".npz"
    scaler_exists = os.path.exists(scaler_path)
    feature_names_exists = os.path.exists(feature_names_path)
    print("\nScaler and feature name files are automatically selected based on model_id string after the last _")
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
    """Gets user choice on whether to predict cost savings for one shipment or time window probabilities for batch.

    Redirects user according to choice.

    Args:
        model (obj): Model to be used for prediction.
        scaler (obj): Scaler used for transforming continuous values.
        feature_names (npz obj): Numpy dict with feature names and dummified feature names.
    """
    # Display menu
    prediction_choice = questionary.select(
        "Which prediction to run?",
        choices=['Predict cost savings for one shipment',
                 'Predict delivery time window probability for batch of shipments',
                 'Go back to main menu'],
        style=blue).ask()
    if prediction_choice == 'Predict cost savings for one shipment':
        predict_one(model, scaler, feature_names)
    elif prediction_choice == 'Predict delivery time window probability for batch of shipments':
        predict_batch(model, scaler, feature_names)
    elif prediction_choice == 'Go back to main menu':
        load_main_menu()
    get_prediction()


def predict_one(model, scaler, feature_names):
    """Predicts cost savings for one shipment. Gets user input for shipment parameters.

    Invokes delivery_prediction_predict.predict_one_cost_savings() for prediction.
    Lastly, display menu to user after prediction.

    Args:
        model (obj): Model to be used for prediction.
        scaler (obj): Scaler used for transforming continuous values.
        feature_names (npz obj): Numpy dict with feature names and dummified feature names.

    """
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
    """Predicts time window probabilities for batch of shipments.

    Prompts user to select input CSV for prediction.
    Loads input CSV and invokes delivery_prediction_predict.predict_batch() for prediction.
    Lastly, display menu to user after prediction.

    Args:
        model (obj): Model to be used for prediction.
        scaler (obj): Scaler used for transforming continuous values.
        feature_names (npz obj): Numpy dict with feature names and dummified feature names.

    """
    # Get input csv files as list
    csv_data_files = []
    for idx, file_path in enumerate(glob.glob(paths.data_delivery_prediction_input_dir + "/*")):
        csv_data_files.append(file_path)
    # Ask for user choice
    batch_data_path = questionary.select(
        "Which csv file to use?",
        choices=csv_data_files,
        style=blue).ask()
    print("\nLoading data file as dataframe...")
    df = pd.read_csv(batch_data_path, dtype=str)
    print("Data file loaded.\n")
    delivery_prediction_predict.predict_batch(df, model, scaler, feature_names)
    # Display menu
    while True:
        choice_after = questionary.select(
            "---Predict cost savings for one shipment---",
            choices=['Predict delivery time window probability for another batch of shipments',
                     'Predict cost savings for one shipment',
                     'Load another model for prediction',
                     'Go back to main menu',
                     'Exit'],
            style=blue).ask()
        if choice_after == 'Predict delivery time window probability for another batch of shipments':
            predict_batch(model, scaler, feature_names)
        elif choice_after == 'Predict cost savings for one shipment':
            predict_one(model, scaler, feature_names)
        elif choice_after == 'Load another model for prediction':
            load_delivery_prediction_menu()
        elif choice_after == 'Go back to main menu':
            load_main_menu()
        elif choice_after == 'Exit':
            sys.exit()


def load_extract_menu():
    """Prompts user for input on data extraction and invokes extract.batch_query().

    """
    extract_data_choice = prompt(menu_options.extract_data_questions, style=blue)
    records = extract.batch_query(extract_data_choice['start_date'],
                                  extract_data_choice['end_date'],
                                  extract_data_choice['sample_size'])
    extract.store(records, extract_data_choice['sample_size'])
    # Display menu
    while True:
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
    """Prompts user to select extracted data file.

    Loads extracted data file and invokes calculate_matrix()
    """
    # Load extracted data
    extracted_data = load_extracted_data()
    calculate_matrix(extracted_data)


def calculate_matrix(extracted_data):
    """Calculates matrix based on user input weights and extracted data.

    Invokes benchmarking_calc.create_similarity_score_matrix() to create similarity matrix.
    Invokes benchmarking_calc.create_customer_KPI_database() to create KPI database.
    Lastly, display menu to user after calculation.

    Args:
        extracted_data (pandas df obj): Pandas dataframe containing data extracted from 71lbs database

    """
    # Get user input for metric weights
    metric_weights = {
        'weight_vs': 1 / 6,
        'weight_vpz': 1 / 6,
        'weight_vpm': 1 / 6,
        'weight_ws': 1 / 6,
        'weight_wpz': 1 / 6,
        'weight_wpm': 1 / 6
    }
    EVEN = questionary.confirm("Create a new similarity matrix with even metric weights?").ask()
    if not EVEN:
        print("\nPlease enter weight for each metric. Please make sure that sum of all 6 weights is approximately = 1")
        metric_weights = prompt(menu_options.benchmarking_metric_weights, style=blue)
        metric_weights_sum = sum(metric_weights.values())
        print("\nRecalibrating weights...")
        for weight in metric_weights:
            metric_weights[weight] = metric_weights[weight] / metric_weights_sum
    print("\nWeights:", metric_weights, "\n")
    print("\nCreating similarity score matrix...")
    benchmarking_calc.create_similarity_score_matrix(extract_data=extracted_data, input_weights=metric_weights)
    benchmarking_calc.create_customer_KPI_database(arg_df=extracted_data)
    while True:
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
    """Prompts user to preprocess data and train model after extraction.

    Complete steps to train a model: Extract data -> Preprocess data -> Train model
    If extraction is needed, user is redirected to extract data from database.
    If preprocessing is needed, invoke preprocess() before train()
    If user only wishes to change model parameters for existing preprocessed data, only invoke train()
    Lastly, display menu after training.

    """
    print("\nComplete steps to train a model: Extract data -> Preprocess data -> Train model")
    print(
        "\nChoose `Option 1` if you have not extracted a new dataset. Alternatively, choose `Option 3` from the Main Menu.")
    print("\nChoose `Option 2: Preprocess data -> Train model` if you have already extracted a new dataset")
    print("\nOnly choose `Option 3: Train model` if you have trained a model previously,",
          "\nand only wish to try again on the same dataset with different hyperparameters\n")
    train_steps = questionary.select(
        "Select steps to execute",
        choices=['Extract data',
                 'Preprocess data -> Train model',
                 'Train model'],
        style=blue).ask()
    if train_steps == 'Extract data':
        load_extract_menu()
    elif train_steps == 'Preprocess data -> Train model':
        extracted_data = load_extracted_data()
        datadict, model_id = preprocess(extracted_data)
        train(datadict, model_id)
    elif train_steps == 'Train model':
        # Choose data_dict
        datadict_files = []
        for idx, file_path in enumerate(glob.glob(paths.data_delivery_prediction_datadict_dir + "/*")):
            datadict_files.append(file_path)
        datadict_path = questionary.select(
            "Which preprocessed data file to use?",
            choices=datadict_files,
            style=blue).ask()
        datadict = np.load(datadict_path, allow_pickle=True)
        # Extract model_id from data_dict
        model_id = datadict_path.split('_', )[-1].split('.', )[0]
        print("Model_id:", model_id)
        train(datadict, model_id)
    # Display menu
    while True:
        choice_after = questionary.select(
            "---Train delivery prediction model---",
            choices=['Train another model on same data but with different parameters',
                     'Go back to main menu',
                     'Exit'],
            style=blue).ask()
        if choice_after == 'Train another model on same data but with different parameters':
            train(datadict, model_id)
        elif choice_after == 'Go back to main menu':
            load_main_menu()
        elif choice_after == 'Exit':
            sys.exit()


def preprocess(extracted_data):
    """Invokes delivery_prediction_preprocess.preprocess() to preprocess data.

    Args:
        extracted_data (pandas df obj): Pandas dataframe containing data extracted from 71lbs database

    Returns:
        datadict (dict): Dictionary of numpy arrays containing preprocessed train and test data.
        model_id (str): Timestamp used to identify model, scaler and feature names files

    """
    datadict, model_id = delivery_prediction_preprocess.preprocess(extracted_data)
    return datadict, model_id


def train(datadict, model_id):
    """Prompts user to select the most important random forest model parameters.

    Invokes delivery_prediction_train.train() to train model.

    Args:
        datadict (Numpy object): Numpy dictionary containing preprocessed train and test datasets.
        model_id (str): Timestamp used to identify model, scaler and feature names files

    """
    # Prompt for user input for parameters e.g. estimators, depth
    n_estimators = questionary.text(
        "Enter number of trees to use for Random Forest training",
        default="25",
        validate=menu_options.NumberValidator,
        style=blue).ask()
    max_depth = questionary.text(
        "Enter maximum depth for Random Forest training",
        default="50",
        validate=menu_options.NumberValidator,
        style=blue).ask()
    delivery_prediction_train.train(datadict, model_id, n_estimators, max_depth)


def load_extracted_data():
    """ Gets user choice on extracted data file and loads file for matrix calculation/prediction.

    """
    # Display menu
    data_extracted = questionary.select(
        "!! You will need to extract a new dataset before proceeding !!",
        choices=['I have already extracted data',
                 'I need to extract new data',
                 'Go back to main menu'],
        style=blue).ask()
    # Load model
    if data_extracted == 'I have already extracted data':
        # Get extracted files as list
        extracted_data_files = []
        for idx, file_path in enumerate(glob.glob(paths.data_extracted_dir + "/*")):
            extracted_data_files.append(file_path)
        # Ask for user choice
        extracted_data_path = questionary.select(
            "Which extracted data file to use?",
            choices=extracted_data_files,
            style=blue).ask()
        print("\nLoading data file...")
        extracted_data = joblib.load(extracted_data_path)
        print("Data file loaded.\n")
        return extracted_data
    elif data_extracted == 'I need to extract new data':
        load_extract_menu()
    elif data_extracted == 'Go back to main menu':
        load_main_menu()


print("Welcome to the 71bs dashboard")
load_main_menu()
