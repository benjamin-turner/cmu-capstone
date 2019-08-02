"""
Delivery Prediction Predict Script

This script contains the functions to preprocess user console input,
predict time windows using saved models and estimate cost savings for
a single shipment.

This file should be imported as a module and contains the following
function that is used in main.py:

    * predict_one_cost_savings - preprocess, predict, format cost savings
    for one shipment
"""
import os
import time
from datetime import datetime
from datetime import timedelta
import joblib
import numpy as np
import pandas as pd
import pgeocode
import shippo
from tabulate import tabulate
from uszipcode import SearchEngine
import credentials, utilities, paths


def get_distance(zipcode1, zipcode2):
    """
    Gets geodesic distance between 2 given zipcodes
    https://pgeocode.readthedocs.io/en/latest/overview.html

    Args:
            zipcode1 (str): sender 5-digit zipcode
            zipcode2 (str): recipient 5-digit zipcode

    Returns:
            float: distance between zipcodes in miles
    """
    dist = pgeocode.GeoDistance('us')
    return dist.query_postal_code(zipcode1, zipcode2)


def get_zip_details(zipcode1, search):
    """
    Get population, population density, no. housing units
    and state for given zipcode
    https://pgeocode.readthedocs.io/en/latest/overview.html

    Args:
            zipcode1 (str): target 5-digit zipcode
            search (obj): uszipcode search object

    Returns:
            pop (int): population
            pop_density (int): population density
            housing_units (int): number of housing units
            state (str): 2-letter US state code
    """
    zipcode = search.by_zipcode(zipcode1)
    pop = zipcode.population
    pop_density = zipcode.population_density
    housing_units = zipcode.housing_units
    state = zipcode.state
    # Returns 0 if not found. If NaN, program will encounter error
    # during transformation.
    return pop or 0, pop_density or 0, housing_units or 0, state or 0


def get_date_details(shipment_date):
    """
    Gets month, week of year, and day of week from date

    Args:
           shipment_date (str): date in YYYY-MM-DD format

    Returns:
            week_number (int): week of year
            day_of_week (int): day of week
            month (int): month of year
    """
    shipment_date_parsed = pd.Series(datetime.strptime(shipment_date, '%Y-%m-%d'))
    week_number = shipment_date_parsed.dt.week.values[0]
    day_of_week = shipment_date_parsed.dt.dayofweek.values[0]
    month = shipment_date_parsed.dt.month.values[0]
    return week_number, day_of_week, month


def get_msa_details(sender_zip, recipient_zip):
    """
    Converts sender zipcode and recipient zipcode to
    Metropolitan Statistical Area features

    Micropolitan Statistical Areas
    A MICRO is simply a small CBSA, i.e., a county or
    counties with an urbanized core of 10,000 but fewer
    than 50,000 in population. Outlying areas included are,
    again, defined by commuting patterns. As of November
    2004, according to the Census Bureau, there were 575
    MICROs in the U.S. and five in Puerto Rico.

    Metropolitan Statistical Areas
    An MSA has an urbanized core of minimally 50,000
    population and includes outlying areas determined
    by commuting measures. In 2004, the U.S. had 362
    MSAs and eight in Puerto Rico.

    Args:
            sender_zip (str): sender 5-digit zipcode
            recipient_zip (str): recipient 5-digit zipcode

    Returns:
            sender_in_MSA (bool): True if sender zipcode is in MSA

            recipient_in_MSA (bool): True if recipient zipcode is
            in MSA

            same_MSA (book): True if sender and recipient zipcode
            are in same MSA
    """
    # Read external CSV that contains zipcode mapping to MSA codes
    zipcode_to_msa_df = pd.read_csv(paths.data_delivery_prediction_msa_dir_cmu, dtype=object)
    zipcode_to_msa_df.columns = ['zipcode', 'state', 'msa_num', 'county_num', 'msa_name']
    zip_msa_num_dict = zipcode_to_msa_df.set_index('zipcode')['msa_num'].to_dict()
    zip_msa_name_dict = zipcode_to_msa_df.set_index('zipcode')['msa_name'].to_dict()
    sender_zip = str(sender_zip)
    recipient_zip = str(recipient_zip)
    # Search for sender zipcode in MSA mapping
    if sender_zip in zip_msa_num_dict:
        sender_msa_num = zip_msa_num_dict[sender_zip]
        msa_name = zip_msa_name_dict[sender_zip]
        if 'MSA' in msa_name:
            sender_in_msa = 1
        else:
            sender_in_msa = 0
    else:
        sender_msa_num = "N/A"
        sender_in_msa = 0
    # Search for recipient zipcode in MSA mapping
    if recipient_zip in zip_msa_num_dict:
        recipient_msa_num = zip_msa_num_dict[recipient_zip]
        msa_name = zip_msa_name_dict[recipient_zip]
        if 'MSA' in msa_name:
            rec_in_msa = 1
        else:
            rec_in_msa = 0
    else:
        recipient_msa_num = "N/A"
        rec_in_msa = 0
    # Check if sender zipcode and recipient zipcode have the same MSA code
    if sender_msa_num == recipient_msa_num:
        same_msa = 1
    else:
        same_msa = 0
    return sender_in_msa, rec_in_msa, same_msa


def get_shippo_details(shipper, weight, sender_zip, sender_state, recipient_zip, recipient_state):
    """
    Retrieve shipment delivery rates and zones from Shippo API
    for UPS and FedEx based on input.

    Rates are used to calculate cost savings between different
    service types and ground service.

    Shippo API has a call limit of 400
    calls per hour. Function calls API twice, once to create
    shipment, and another time to get shipment rates.
    https://goshippo.com/docs/reference

    Args:
            shipper (str): shipper name. only accepts FedEx or UPS.
            weight (float): shipment weight
            sender_zip (str): sender 5-digit zipcode
            sender_state (str): sender 2-letter state code
            recipient_zip (str): recipient 5-digit zipcode
            recipient_state (str): recipient 2-letter state code

    Returns:
            df (pandas dataframe obj): dataframe with service type,
            ship cost, ground cost, cost savings, scheduled window
            numbers, scheduled time windows as columns

            zone (int): shipper classified zone. Default=5.
            If zone is found, replace default.
            Shippo API does not return UPS zone.
            FedEx zone assumed for all UPS shipments.
    """
    shipper = str.lower(shipper)
    # Shippo API key is stored in credentials
    shippo.api_key = credentials.shippo_test_key
    address_from = {"state": sender_state, "zip": sender_zip, "country": "US"}
    address_to = {"state": recipient_state, "zip": recipient_zip, "country": "US"}
    # Create parcel with true weight and dummy dimensions
    parcel = {
        "length": "5", "width": "5", "height": "5", "distance_unit": "in",
        "weight": weight, "mass_unit": "lb"
    }
    # Create shipment with parcel
    shipment = shippo.Shipment.create(
        address_from=address_from,
        address_to=address_to,
        parcels=[parcel],
        asynchronous=False
    )
    # Retrieve shipment ID created by Shippo
    shippo_id = shippo.Shipment.retrieve(shipment.object_id)['object_id']
    # Get rates in JSON format with shipment ID
    rates = shippo.Shipment.get_rates(shippo_id)
    # Store rates information in results dict
    results = {}
    # Set default zone to 5. Only used if zone is not found.
    zone = 5
    # Store shipper service types as keys and cost as values
    for i in rates['results']:
        results[i['servicelevel']['token']] = i['amount']
        # Replace default zone with FedEx ground zone.
        # UPS zones are not available through Shippo API
        if i['servicelevel']['token'] == "fedex_ground":
            zone = i['zone']
    # Return None if no results are found
    if not bool(results):
        print("No results from Shippo. Please try again.")
        return None, None
    else:
    # Create shipper service dict with service types as keys and
    # scheduled window numbers as values
        fedex_services_dict = {
            'fedex_first_overnight': 1,
            'fedex_priority_overnight': 2,
            'fedex_standard_overnight': 3,
            'fedex_2_day_am': 5,
            'fedex_2_day': 6,
            'fedex_express_saver': 8
        }
        ups_services_dict = {
            'ups_next_day_air_early_am': 1,
            'ups_next_day_air': 2,
            'ups_next_day_air_saver': 3,
            'ups_second_day_air_am': 5,
            'ups_second_day_air': 6,
            'ups_3_day_select': 8
        }
        # Filter dict to use based on shipper
        if shipper == 'ups':
            chosen_dict = ups_services_dict
        elif shipper == 'fedex':
            chosen_dict = fedex_services_dict
        services = []
        ship_costs = []
        ground_costs = []
        cost_savings = []
        scheduled_windows = []
        scheduled_window_nos = []
        # Store ground delivery cost
        try:
            if shipper == 'ups':
                ground_cost_ = float(results['ups_ground'])
            elif shipper == 'fedex':
                ground_cost_ = float(results['fedex_ground'])
        except KeyError:
            print(shipper, "does not send to the zipcode provided. Please try again.")
        # Load dict that maps time window numbers to time windows
        windows_cmu = joblib.load(paths.windows_cmu)
        # Append results to filtered shipper service types
        for i in chosen_dict.keys():
            if i in results.keys():
                services.append(i)
                ship_costs.append(results[i])
                ground_costs.append(ground_cost_)
                cost_savings.append(float(results[i]) - ground_cost_)
                scheduled_window_no = chosen_dict[i]
                scheduled_window_nos.append(scheduled_window_no)
                scheduled_windows.append(windows_cmu[scheduled_window_no])
        # Create rates dataframe to display rates information and scheduled time windows
        headers = ['service','ship_cost', 'ground_cost', 'cost_saving', 'scheduled_window_no', 'scheduled_window']
        df = pd.DataFrame(list(zip(services, ship_costs, ground_costs, cost_savings, scheduled_window_nos, scheduled_windows)), columns=headers)
        return df, zone


def preprocess_one(shipment_date, shipper, std_weight, sender_zip, recipient_zip, scaler):
    """
    Preprocesses input to create features that model will
    predict on.

    Args:
            shipment_date (str): shipment date in YYYY-MM-DD format
            shipper (str): shipper name. only accepts FedEx or UPS.
            std_weight (float): shipment weight
            sender_zip (str): sender 5-digit zipcode
            recipient_zip (str): recipient 5-digit zipcode
            scaler (str): file path to scaler based on model

    Returns:
            test (np array object): input for model prediction.
            Contains 239 dummified features.

            rates_df (pandas dataframe obj): dataframe with
            service type, ship cost, ground cost, cost savings,
            scheduled window numbers, scheduled time windows as
            columns
    """
    print("Preprocessing input...")
    start_time = time.time()
    # Get datetime features
    week_number, day_of_week, month = get_date_details(shipment_date)
    # Get sender_in_msa and recipient_in_MSA and same_msa booleans
    sender_in_msa, rec_in_msa, same_msa = get_msa_details(sender_zip, recipient_zip)
    # Get distance
    distance = round(get_distance(sender_zip, recipient_zip), 5)
    # Get population, density, no. houses, state code for recipient and sender
    search = SearchEngine()
    recipient_pop, recipient_pop_density, recipient_houses, recipient_state = get_zip_details(recipient_zip, search)
    sender_pop, sender_pop_density, sender_houses, sender_state = get_zip_details(sender_zip, search)
    # Get rates dataframe and zone
    rates_df, zone = get_shippo_details(shipper, std_weight, sender_zip, sender_state, recipient_zip, recipient_state)
    # Load feature names saved by CMU
    feature_names = np.load(paths.data_delivery_prediction_features_dir_cmu, allow_pickle=True)
    # Create empty dataframe with correct columns that model was trained on
    df = pd.DataFrame(columns=feature_names['feature_names'])
    # Add new row into df with test data
    df.loc[0] = [shipper, std_weight, zone,
                 sender_state, recipient_state, distance, sender_pop, sender_pop_density,
                 sender_houses, recipient_pop, recipient_pop_density, recipient_houses, same_msa,
                 sender_in_msa, rec_in_msa, week_number, day_of_week, month]
    # Define categorical and float columns for one-hot-encoding purposes
    cat_cols = ['shipper', 'zone', 'week_number', 'day_of_week',
                'sender_state', 'recipient_state', 'month']
    float_cols = ['std_weight', 'distance', 'sender_pop', 'sender_pop_density', 'sender_houses', 'recipient_pop', 'recipient_pop_density',
                  'recipient_houses']
    df[cat_cols] = df[cat_cols].astype('category')
    df[float_cols] = df[float_cols].astype('float64')
    # Create one-hot-encoded variables from categorical columns
    df = pd.get_dummies(df)
    # Create empty dataframe in same shape as the one used in model, fill with 0s
    df_full = pd.DataFrame(columns=feature_names['feature_names_dummified'])
    # Execute a right join to align test dataframe with dataframe that model was trained on
    df, df_full = df.align(df_full, join='right', axis=1, fill_value=0)
    # Convert dataframe to numpy array for prediction
    test = df.loc[0].values
    # Scale data with saved min-max scaler
    test = test.reshape(1, -1)
    scaler = joblib.load(scaler)
    test = scaler.transform(test)
    utilities.print_elapsed_time(start_time)
    print(" ")
    print("preprocess")
    print(test, rates_df)
    return test, rates_df


def predict_time_windows(test, model):
    """
    Predicts time window probability distribution with
    saved model for test input.

    Args:
            test (np array object): input for model prediction.
            Contains 239 dummified features.
            model (str): file path to model that user selected.


    Returns:
            pred (int): predicted ground delivery time window no.
            based on mode of predicted probability distribution.

            pred_proba (np array object): predicted probability
            distribution across time windows. Sums to 1.
            Shape = (no. time windows, no. of test shipments).
    """
    start_time = time.time()
    print("Loading trained model and predicting time windows...")
    # disabled for demo
    # model = joblib.load(model)
    pred = model.predict(test)
    pred_proba = model.predict_proba(test)
    utilities.print_elapsed_time(start_time)
    print(" ")
    print("predict")
    print(pred, pred_proba)
    return pred, pred_proba


def format_cost_savings(pred, pred_proba, rates_df):
    """
    Formats dataframe for output

    Args:
            pred (int): predicted ground delivery time window no.
            based on mode of predicted probability distribution.

            pred_proba (np array object): predicted probability
            distribution across time windows. Sums to 1.
            Shape of (no. time windows, no. of test shipments).

            rates_df (pandas dataframe obj): dataframe with service
            type, ship cost, ground cost, cost savings,
            scheduled window numbers, scheduled time windows as
            columns

    Returns:
            Pandas dataframe object: dataframe with service type,
            ship cost, ground cost, cost savings, scheduled time
            windows, predicted time windows, cumulative
            probability as columns
    """
    start_time = time.time()
    print("Calculating cost savings and probabilities...")
    # Load dict that maps time window numbers to time windows
    windows_cmu = joblib.load(paths.windows_cmu)
    # Insert predicted time window into dataframe
    # windows_cmu maps time window number to corresponding time window
    try:
        rates_df['pred_ground_window'] = windows_cmu[pred]
    except TypeError:
        print("Shipper did not return any results")
        pass
    # Insert predicted probability distribution array object into each cell in column
    rates_df['pred_ground_window_pdf'] = 0
    rates_df['pred_ground_window_pdf'] = rates_df['pred_ground_window_pdf'].astype(object)
    for i in range(len(rates_df)):
        rates_df.at[i, 'pred_ground_window_pdf'] = pred_proba
    # For each probability distribution array object, slice the array with scheduled window number
    # Sum of the sliced array is the predicted cumulative probability of shipment arriving before or in scheduled window
    rates_df['pred_probability'] = rates_df.apply(lambda x: np.sum(x['pred_ground_window_pdf'][:x['scheduled_window_no'] + 1]), axis=1)
    # Converts fraction into percentage for display
    rates_df['pred_probability'] = pd.Series(["{0:.2f}%".format(val * 100) for val in rates_df['pred_probability']], index=rates_df.index)
    # Drop columns that are not needed for display
    rates_df = rates_df.drop(columns=['pred_ground_window_pdf', 'scheduled_window_no'])
    utilities.print_elapsed_time(start_time)
    print(" ")
    return rates_df


def predict_one_cost_savings(shipment_date, shipper, weight, sender_zip, recipient_zip,
                             scaler=paths.scaler_cmu, model=paths.model_cmu):
    """
    Combines preprocessing, prediction, output formatting
    functions to create dataframe for output to user

    Args:
            shipment_date (str): shipment date in YYYY-MM-DD format
            shipper (str): shipper name. only accepts FedEx or UPS.
            std_weight (float): shipment weight
            sender_zip (str): sender 5-digit zipcode
            recipient_zip (str): recipient 5-digit zipcode
            scaler (str): file path to scaler corresponding to
            selected model. Default: CMU scaler

            model (str): file path to model that user selected in
            console. Default: CMU model

    Returns:
            Pandas dataframe object: dataframe with service type,
            ship cost, ground cost, cost savings, scheduled time
            windows, predicted time windows, cumulative probability
            as columns
    """
    # Create model_id using current timestamp
    model_id = utilities.create_id()
    # Preprocess user input
    preprocessed_input, rates_df = preprocess_one(shipment_date, shipper, weight, sender_zip, recipient_zip, scaler)
    # Predict based on preprocessed input
    pred, pred_proba = predict_time_windows(preprocessed_input, model)
    # Merge results into dataframe for output
    cost_savings_df = format_cost_savings(pred[0], pred_proba[0], rates_df)
    if cost_savings_df is None:
        print("Unable to predict. Please check input.")
        return None
    # Save output to CSV with model_id
    output_path = os.path.join(paths.output_delivery_prediction_dir, model_id)
    cost_savings_df.to_csv(output_path+"_predict_one.csv")
    # Print output using tabulate package for formatting
    print(tabulate(cost_savings_df, headers='keys', showindex=False, floatfmt=".2f", tablefmt='psql',
                   colalign=['left', 'center', 'center', 'center', 'center', 'center', 'center']))
    print("Results saved to " + output_path + "_predict_one.csv")
    return cost_savings_df


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    # Predict for one instance
    predict_one_cost_savings("2019-07-09", "ups", 9, 91724, 15206)


