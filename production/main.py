# coding: utf8

from __future__ import print_function, unicode_literals
import datetime
import joblib
import os
import sys
from pprint import pprint
from PyInquirer import prompt
from examples import custom_style_3
# Import internal modules
import delivery_prediction_predict
import menu_options
import paths
import extract


def load_main_menu():
    main_menu_choice = prompt(menu_options.main_menu, style=custom_style_3)
    user_choice = menu_options.main_menu_options.index(main_menu_choice['main_menu']) + 1
    if user_choice == 1:
        # TODO: Shiv code for benchmarking
        sys.exit()
    if user_choice == 2:
        load_delivery_prediction_menu()
    if user_choice == 3:
        load_extract_menu()
    if user_choice == 4:
        # TODO: Shiv code for preprocessing and calculation matrix
        sys.exit()
    if user_choice == 5:
        # TODO: Royce code for preprocessing and training
        sys.exit()
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
    load_extract_questions()
    while True:
        extract_menu_choice = prompt(menu_options.extract_menu, style=custom_style_3)
        extract_menu_choice_no = menu_options.extract_menu_options.index(
            extract_menu_choice['extract_menu']) + 1
        if extract_menu_choice_no == 1:
            load_extract_questions()
        elif extract_menu_choice_no == 2:
            load_main_menu()
        else:
            sys.exit()


def load_extract_questions():
    extract_data_answers = prompt(menu_options.extract_data_questions, style=custom_style_3)
    extract.extract_data(extract_data_answers['start_date'],
                         extract_data_answers['end_date'],
                         extract_data_answers['sample_size'])

print("Welcome to the 71bs dashboard")
# For demo purposes, we preload the models
# print("Preloading model...")
# preloaded_model = joblib.load(paths.model_cmu)
preloaded_model = "FILLER"
# print("Model preloaded.")

load_main_menu()






