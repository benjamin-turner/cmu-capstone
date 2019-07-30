import datetime
import glob
import sys
import joblib
import zipcodes

import delivery_prediction_predict
import paths


def get_user_choice_int(beg_int, end_int):
    user_choice = 0
    while user_choice not in range(beg_int, end_int+1):
        try:
            user_choice = input("Please enter your choice (" + str(beg_int) + "-" + str(end_int) + "):")
            user_choice = int(user_choice)
        except ValueError:
            print("You did not enter an integer")
    print("---")
    return user_choice

def get_date():
    valid = False
    while not valid:
        user_date = input("Enter shipment date in YYYY-MM-DD format: ")
        try:
            datetime.datetime.strptime(user_date, '%Y-%m-%d')
            valid = True
        except ValueError:
            print("Date input is incorrect, please try again.")
            valid = False
    print("---")
    return user_date

def load_main_menu():
    print("1. Run benchmarking")
    print("2. Run delivery prediction")
    print("3. Calculate similarity matrix")
    print("4. Train delivery prediction model")
    print("5. Exit")
    user_choice = get_user_choice_int(1, 5)
    if user_choice == 1:
        # TODO: Shiv code for benchmarking
        sys.exit()
    if user_choice == 2:
        load_delivery_prediction_menu()
    if user_choice == 3:
        # TODO: Shiv code for calculating matrix
        sys.exit()
    if user_choice == 4:
        # TODO: Royce code for training
        sys.exit()
    if user_choice == 5:
        sys.exit()


def load_delivery_prediction_menu():
    print("Run delivery prediction")
    print("1. Predict cost savings for one shipment")
    print("2. Predict cost savings for batch")
    print("3. Predict delivery time window probability for batch")
    print("4. Go back to main menu")
    prediction_choice = get_user_choice_int(1, 3)
    # Get user to choose model to use for prediction
    print("Please choose model to use for prediction: ")
    models = []
    for idx, model in enumerate(glob.glob(paths.model_dir + "/*.pkl.z")):
        models.append(model)
        print(f"{idx + 1}: {model}")
    model_choice = get_user_choice_int(1, len(models))
    model = models[model_choice - 1]
    print(f"Using model: {model} \n---")
    # Get user to choose scaler to use for prediction
    print("Please choose scaler to use for prediction: ")
    scalers = []
    for idx, scaler in enumerate(glob.glob(paths.model_scaler_dir + "/*.pkl.z")):
        scalers.append(scaler)
        print(f"{idx + 1}: {scaler}")
    scaler_choice = get_user_choice_int(1, len(scalers))
    scaler = scalers[scaler_choice - 1]
    print(f"Using scaler: {scaler} \n---")
    # If user predicts 1 shipment
    if prediction_choice == 1:
        get_input_predict_one(scaler, model)
        while True:
            print("---")
            print("1. Predict one shipment")
            print("2. Go back to main menu")
            print("3. Exit")
            predict_one_choice = get_user_choice_int(1, 3)
            if predict_one_choice == 1:
                get_input_predict_one(scaler, model)
            elif predict_one_choice == 2:
                load_main_menu()
            else:
                sys.exit()
    # TODO: predict batch


def get_input_predict_one(scaler, model):
    # Get user input for shipment_date, shipper, weight, sender_zip, recipient_zip,
    shipment_date = get_date()
    weight = input("Enter shipment weight in lbs: ")
    print("---")
    sender_zip = get_zipcode("sender")
    recipient_zip = get_zipcode("recipient")
    print("Please choose shipper")
    print("1. ups")
    print("2. fedex")
    shipper_choice = get_user_choice_int(1, 2)
    if shipper_choice == 1:
        shipper = "ups"
    else:
        shipper = "fedex"
    # Run prediction
    df = delivery_prediction_predict.predict_one_cost_savings(shipment_date, shipper, weight, sender_zip, recipient_zip, scaler, preloaded_model)
    return df


def get_zipcode(type):
    valid = False
    while not valid:
        user_zipcode = input("Enter " + type + " zipcode: ")
        if user_zipcode.isdigit() and len(user_zipcode) == 5:
            valid = zipcodes.is_real(user_zipcode)
    print("---")
    return user_zipcode

print("Welcome to the 71bs dashboard")
# For demo purposes, we preload the models
print("Preloading model...")
preloaded_model = joblib.load(paths.model_cmu)
print("Model preloaded.")

load_main_menu()






