"""
Delivery Prediction Predict Script

This script contains the functions to predict cost savings for one shipment, and time window probabilities for
batches of shipments stored in a CSV file.

To note: For CSV batch shipment input, CSV file must be stored in delivery_prediction/input folder and
contain the following column names with fixed format:
shipment_date (str): YYYY-MM-DD format
sender_zip (str): String representation of 5-digit zipcode
recipient_zip (str): String representation of 5-digit zipcode
weight (str): Shipment weight in pounds
shipper (str): ups or fedex
service_type (str): Shipper service type with fixed format. Refer to delivery_prediction/input/batch_sample_cmu.csv
zone (int): 2 to 8

This file should be imported as a module and contains the following function that is used in main.py:
    * predict_one_cost_savings - preprocess, predict, format cost savings for predicting cost savings with one shipment
    * predict_batch - preprocess, predict and format output for predicting batch time window probabilities

"""
import datetime
import os
import time
import joblib
import numpy as np
import pandas as pd
import pgeocode
import shippo
import zipcodes
from tabulate import tabulate
from tqdm import tqdm
from uszipcode import SearchEngine
import credentials, paths, utilities, delivery_prediction_preprocess


def get_distance(zipcode1, zipcode2):
    """Get geodesic distance between 2 given zipcodes
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
    """Get population, population density, no. housing units and state for given zipcode
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
    # Returns 0 if not found. If NaN, program will encounter error during transformation.
    return pop or 0, pop_density or 0, housing_units or 0, state or 0


def get_date_details(shipment_date):
    """Get month, week of year, and day of week from date

    Args:
       shipment_date (str): Date in YYYY-MM-DD format

    Returns:
        week_number (int): Week of year
        day_of_week (int): Day of week
        month (int): Month of year
    """
    shipment_date_parsed = pd.Series(datetime.datetime.strptime(shipment_date, '%Y-%m-%d'))
    week_number = shipment_date_parsed.dt.week.values[0]
    day_of_week = shipment_date_parsed.dt.dayofweek.values[0]
    month = shipment_date_parsed.dt.month.values[0]
    return week_number, day_of_week, month


def get_msa_details(sender_zip, recipient_zip):
    """Converts sender zipcode and recipient zipcode to Metropolitan Statistical Area features

    Micropolitan Statistical Areas
    A MICRO is simply a small CBSA, i.e., a county or counties with an urbanized core of 10,000 but fewer than 50,000.
    Outlying areas included are again, defined by commuting patterns. As of November 2004,
    according to the Census Bureau, there were 575 MICROs in the U.S. and five in Puerto Rico.

    Metropolitan Statistical Areas
    An MSA has an urbanized core of minimally 50,000 population and includes outlying areas determined
    by commuting measures. In 2004, the U.S. had 362 MSAs and eight in Puerto Rico.

    Args:
        sender_zip (str): Sender 5-digit zipcode
        recipient_zip (str): Recipient 5-digit zipcode

    Returns:
        sender_in_MSA (bool): True if sender zipcode is in MSA
        recipient_in_MSA (bool): True if recipient zipcode is in MSA
        same_MSA (book): True if sender and recipient zipcode are in same MSA
    """
    # Convert to string
    sender_zip = str(sender_zip)
    recipient_zip = str(recipient_zip)
    # Read external CSV that contains zipcode mapping to MSA codes
    zipcode_to_msa_df = pd.read_csv(paths.data_delivery_prediction_zip_to_msa_cmu, dtype=object)
    # If MSA Name contains 'MSA', zipcode is in a MSA i.e. bool=True
    zipcode_to_msa_df['MSA'] = zipcode_to_msa_df['MSA Name'].str.contains(r'MSA')
    # Only keep rows where zipcodes are MSAs
    zipcode_to_msa_df = zipcode_to_msa_df[zipcode_to_msa_df['MSA'].isin([True])]
    # Create pd series for all zipcodes in our df
    s = pd.Series(sender_zip)
    r = pd.Series(recipient_zip)
    # Add columns to df for all zipcodes that are found in zipcode column of zipcode_to_msa_df
    sender_in_msa = s.isin(zipcode_to_msa_df['ZIP CODE']).values[0]
    rec_in_msa = r.isin(zipcode_to_msa_df['ZIP CODE']).values[0]
    zip_to_msa_dict = zipcode_to_msa_df.set_index('ZIP CODE')['MSA No.'].to_dict()
    try:
        if zip_to_msa_dict[sender_zip] == zip_to_msa_dict[recipient_zip]:
            same_msa = True
        else:
            same_msa = False
    except KeyError:
        same_msa = False
    return sender_in_msa, rec_in_msa, same_msa


def get_shippo_details(shipper, weight, sender_zip, sender_state, recipient_zip, recipient_state):
    """Retrieve shipment delivery rates and zones from Shippo API for UPS and FedEx based on input.

    Rates are used to calculate cost savings between different service types and ground service.

    Shippo API has a call limit of 400 calls per hour. Function calls API twice, once to create
    shipment, and another time to get shipment rates.
    Reference: https://goshippo.com/docs/reference

    Args:
        shipper (str): Shipper name. only accepts FedEx or UPS.
        weight (float): Shipment weight
        sender_zip (str): Sender 5-digit zipcode
        sender_state (str): Sender 2-letter state code
        recipient_zip (str): Recipient 5-digit zipcode
        recipient_state (str): Recipient 2-letter state code

    Returns:
        df (pandas dataframe obj): Dataframe with service type, ship cost, ground cost, cost savings, scheduled window
            numbers, scheduled time windows as columns
        zone (int): Shipper classified zone. Default=5.
            If zone is found, replace default. Shippo API does not return UPS zone. FedEx zone assumed for all UPS shipments.
    """
    shipper = str.lower(shipper)
    # Shippo API key is stored in credentials
    shippo.api_key = credentials.shippo_live_key
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
        # Load shipper service dict with service types as keys and scheduled window numbers as values
        fedex_services_dict = joblib.load(paths.fedex_service_types_to_time_window)
        ups_services_dict = joblib.load(paths.ups_service_types_to_time_window)
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
        headers = ['service', 'ship_cost', 'ground_cost', 'cost_saving', 'scheduled_window_no', 'scheduled_window']
        df = pd.DataFrame(
            list(zip(services, ship_costs, ground_costs, cost_savings, scheduled_window_nos, scheduled_windows)),
            columns=headers)
        return df, zone


def preprocess_one(shipment_date, shipper, std_weight, sender_zip, recipient_zip, scaler, feature_names):
    """Preprocesses input to create features that model will predict on.

    Args:
        shipment_date (str): Shipment date in YYYY-MM-DD format
        shipper (str): Shipper name. only accepts FedEx or UPS.
        std_weight (float): Shipment weight
        sender_zip (str): Sender 5-digit zipcode
        recipient_zip (str): Recipient 5-digit zipcode
        scaler (obj): Scaler based on model
        feature_names (npz object): Dictionary of numpy arrays that contain feature names

    Returns:
        test (np array object): Input for model prediction.
        rates_df (pandas dataframe obj): Dataframe with service type, ship cost, ground cost, cost savings,
            scheduled window numbers, scheduled time windows as columns
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
    float_cols = ['std_weight', 'distance', 'sender_pop', 'sender_pop_density', 'sender_houses', 'recipient_pop',
                  'recipient_pop_density',
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
    test = scaler.transform(test)
    utilities.print_elapsed_time(start_time)
    return test, rates_df


def predict_time_windows(test, model):
    """Predicts time window probability distribution with saved model for test input.

    Args:
        test (array): Input for model prediction.
        model (obj): Loaded model for prediction.


    Returns:
        pred (list): List of predicted ground delivery time window number based on
            mode of predicted probability distribution.
        pred_proba (2D array): Predicted probability distribution across time windows.
            Shape = (no. time windows, no. of test shipments).
    """
    start_time = time.time()
    print("Loading trained model and predicting time windows...")
    pred = model.predict(test)
    pred_proba = model.predict_proba(test)
    utilities.print_elapsed_time(start_time)
    return pred, pred_proba


def format_cost_savings(pred, pred_proba, rates_df):
    """Formats dataframe for predict one output

    Args:
        pred (list): List of predicted ground delivery time window number based on
            mode of predicted probability distribution.
        pred_proba (2D array): Predicted probability distribution across time windows.
            Shape = (no. time windows, no. of test shipments).
        rates_df (pandas dataframe obj): Dataframe with service type, ship cost, ground cost, cost savings,
            scheduled window numbers, scheduled time windows as columns

    Returns:
        Pandas dataframe object: Dataframe with service type, ship cost, ground cost, cost savings,
            scheduled time windows, predicted time windows, cumulative probability as columns
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
    rates_df['pred_probability'] = rates_df.apply(
        lambda x: np.sum(x['pred_ground_window_pdf'][:x['scheduled_window_no'] + 1]), axis=1)
    # Converts fraction into percentage for display
    rates_df['pred_probability'] = pd.Series(["{0:.2f}%".format(val * 100) for val in rates_df['pred_probability']],
                                             index=rates_df.index)
    # Drop columns that are not needed for display
    rates_df = rates_df.drop(columns=['pred_ground_window_pdf', 'scheduled_window_no'])
    utilities.print_elapsed_time(start_time)
    print(" ")
    return rates_df


def predict_one_cost_savings(shipment_date, shipper, weight, sender_zip, recipient_zip,
                             model, scaler, feature_names):
    """Combines preprocessing, prediction, output formatting functions to create dataframe for output to user.

    Args:
        shipment_date (str): Shipment date in YYYY-MM-DD format.
        shipper (str): Shipper name. only accepts FedEx or UPS.
        std_weight (float): Shipment weight.
        sender_zip (str): Sender 5-digit zipcode.
        recipient_zip (str): Recipient 5-digit zipcode.
        model (obj): Model to be used for prediction.
        scaler (str): Scaler corresponding to selected model.
        feature_names (npz object): Dictionary of numpy arrays containing feature names.

    Returns:
            Pandas dataframe object: Dataframe with service type, ship cost, ground cost, cost savings,
                scheduled time windows, predicted time windows, cumulative probability as columns
    """
    # Preprocess user input
    preprocessed_input, rates_df = preprocess_one(shipment_date, shipper, weight, sender_zip, recipient_zip, scaler,
                                                  feature_names)
    # Predict based on preprocessed input
    pred, pred_proba = predict_time_windows(preprocessed_input, model)
    # Merge results into dataframe for output
    cost_savings_df = format_cost_savings(pred[0], pred_proba[0], rates_df)
    if cost_savings_df is None:
        print("Unable to predict. Please check input.")
        return None
    # Save output to CSV with model_id
    timestamp = utilities.get_timestamp()
    output_path = os.path.join(paths.output_delivery_prediction_dir, timestamp + "_predict_one.xlsx")
    # Print output using tabulate package for formatting
    print(tabulate(cost_savings_df, headers='keys', showindex=False, floatfmt=".2f", tablefmt='psql',
                   colalign=['left', 'center', 'center', 'center', 'center', 'center', 'center']))

    # Save output to xlsx file
    pred_proba_df = pd.DataFrame(list(pred_proba)).T
    pred_proba_df.rename(columns={pred_proba_df.columns[0]: "Probability"}, inplace=True)
    pred_proba_df['Probability'] = pd.Series(["{0:.2f}%".format(val * 100) for val in pred_proba_df['Probability']],
                                             index=pred_proba_df.index)
    pred_proba_df.index.rename('Time Window', inplace=True)
    with pd.ExcelWriter(output_path) as writer:
        cost_savings_df.to_excel(writer, sheet_name='Cost Savings')
        pred_proba_df.to_excel(writer, sheet_name='Predicted Window Probabilities')
    print("Results saved to " + output_path)
    return cost_savings_df


def add_zip_details(df):
    """Adds zip details retrieved with get_zip_details() to Dataframe

    Args:
        df (pandas dataframe obj): Pandas dataframe that must contain sender_zip and recipient_zip columns

    Returns:
        pandas dataframe obj: Pandas dataframe with new columns sender_pop, sender_pop_density, sender_houses, sender_state
            recipient_pop, recipient_pop_density, recipient_houses, recipient_state
    """
    print(f"Adding zipcode details, {len(df)} iterations expected...")
    search = SearchEngine()
    pop_list_s, pop_density_list_s, houses_list_s, state_list_s = [], [], [], []
    pop_list_r, pop_density_list_r, houses_list_r, state_list_r = [], [], [], []
    for row in tqdm(df.itertuples()):
        pop_s, pop_density_s, houses_s, state_s = get_zip_details(row.sender_zip, search)
        pop_list_s.append(pop_s)
        pop_density_list_s.append(pop_density_s)
        houses_list_s.append(houses_s)
        state_list_s.append(state_s)

        pop_r, pop_density_r, houses_r, state_r = get_zip_details(row.recipient_zip, search)
        pop_list_r.append(pop_r)
        pop_density_list_r.append(pop_density_r)
        houses_list_r.append(houses_r)
        state_list_r.append(state_r)
    df['sender_pop'], df['sender_pop_density'], df['sender_houses'], df[
        'sender_state'] = pop_list_s, pop_density_list_s, houses_list_s, state_list_s
    df['recipient_pop'], df['recipient_pop_density'], df['recipient_houses'], df[
        'recipient_state'] = pop_list_r, pop_density_list_r, houses_list_r, state_list_r
    return df


def add_scheduled_windows(df):
    """Add scheduled time window according to shipper service type

    Args:
        df (pandas dataframe obj): Pandas dataframe that must contain service_type

    Returns:
        pandas dataframe obj: Pandas dataframe with new column scheduled_window_no
    """
    fedex_services_dict = joblib.load(paths.fedex_service_types_to_time_window)
    ups_services_dict = joblib.load(paths.ups_service_types_to_time_window)
    # Merge dictionaries
    services_dict = {**fedex_services_dict, **ups_services_dict}
    df['scheduled_window_no'] = df.apply(lambda x: services_dict[x['service_type']], axis=1)
    return df


def validate_batch(df):
    """Validates input in CSV.

    Please see module documentation for valid input.

    Args:
        df (pandas dataframe obj): Pandas dataframe that must contain service_type

    Returns:
        True if batch passes validation
    """
    print("Validating input...")
    fedex_services_dict = joblib.load(paths.fedex_service_types_to_time_window)
    ups_services_dict = joblib.load(paths.ups_service_types_to_time_window)
    all_ok = True
    for row in df.itertuples():
        # Validate da1te input
        try:
            datetime.datetime.strptime(row.shipment_date, '%Y-%m-%d')
        except ValueError:
            print(
                f"Found incorrect date input `{row.shipment_date}` in row {row.Index}. Please make sure dates are in YYYY-MM-DD format before trying again.")
            all_ok = False
        # Validate zip codes
        ok = False
        if row.sender_zip.isdigit() and len(row.sender_zip) == 5:
            ok = zipcodes.is_real(row.sender_zip) and \
                 (zipcodes.matching(row.sender_zip)[0]['state'] != 'HI') and \
                 (zipcodes.matching(row.sender_zip)[0]['state'] != 'AK')
        if not ok:
            print(
                f"Found invalid sender zipcode `{row.sender_zip}` in row {row.Index}.\nPlease endPlease amend zipcode before trying again.")
            all_ok = False
        # Validate zip codes
        ok = False
        if row.recipient_zip.isdigit() and len(row.recipient_zip) == 5:
            ok = zipcodes.is_real(row.recipient_zip)
        if not ok:
            print(
                f"Found invalid sender zipcode `{row.recipient_zip}` in row {row.Index}. Please amend zipcode before trying again.")
            all_ok = False
        # Validate weight
        try:
            weight = float(row.weight)
        except ValueError:
            print(f"Found invalid weight `{row.weight}` in row {row.Index}. Please amend weight before trying again.")
            all_ok = False
        # Validate shipper
        ok = False
        if row.shipper.lower() in ['fedex', 'ups']:
            ok = True
        if ok == False:
            print(
                f"Found invalid shipper `{row.shipper}` in row {row.Index}. Model only takes fedex and ups now. Please amend shipper before trying again.")
            all_ok = False
        # Validate service_type
        ok = False
        if row.shipper.lower() in ['fedex']:
            if row.service_type.lower() in list(fedex_services_dict.keys()):
                ok = True
        elif row.shipper.lower() in ['ups']:
            if row.service_type.lower() in list(ups_services_dict.keys()):
                ok = True
        if ok == False:
            print(
                f"Found invalid service type `{row.service_type}` in row {row.Index}. Please amend service type before trying again.")
            all_ok = False
        # Validate zone
        ok = False
        try:
            if int(row.zone) in range(2, 9):
                ok = True
            if not ok:
                print(f"Found invalid zone `{row.zone}` in row {row.Index}. ",
                      "Model only takes zones between 2 and 8. Please amend zone before trying again.")
        except ValueError:
            print(f"Zone must be a number")
            all_ok = False
    return all_ok

    """Combines all preprocessing functions. Invoked by main.

    Args:
        extracted_data (pandas df obj): Pandas dataframe containing data extracted from 71lbs database

    Returns:
        datadict (dict): Dictionary of numpy arrays containing preprocessed train and test data.
        model_id (str): Timestamp used to identify model, scaler and feature names files

    """
def preprocess_batch(df, feature_names, scaler):
    """Combines all preprocessing functions to preprocess data for prediction.

    Args:
        df (pandas dataframe obj): Pandas dataframe after validation passed
        feature_names (npz object): Dictionary of numpy arrays containing feature names.
        scaler (obj): Scaler corresponding to selected model.

    Returns:
        df (pandas dataframe obj): Preprocessed pandas dataframe with new features.
        X_test (array): Test data to be used for prediction.
    """
    print("Preprocessing batch...")
    start_time = time.time()
    print("Adding datetime features")
    # Get date time features
    df['shipment_date'] = pd.to_datetime(df['shipment_date'])
    df['week_number'] = df['shipment_date'].dt.week
    df['day_of_week'] = df['shipment_date'].dt.dayofweek
    df['month'] = df['shipment_date'].dt.month
    print("Adding distance...")
    # Get distance
    df['distance'] = delivery_prediction_preprocess.get_distance(df['sender_zip'].values, df['recipient_zip'].values)
    # Get MSA details
    df = delivery_prediction_preprocess.add_MSA_features(df)
    # Get zip details
    df = add_zip_details(df)
    # Add scheduled time window
    print("Adding scheduled time windows")
    df = add_scheduled_windows(df)

    # Decide on columns to keep
    columns_kept = ['shipper', 'weight', 'zone',
                    'sender_state', 'recipient_state', 'distance', 'sender_pop', 'sender_pop_density',
                    'sender_houses', 'recipient_pop', 'recipient_pop_density', 'recipient_houses', 'same_msa',
                    'sender_in_msa', 'rec_in_msa', 'week_number', 'day_of_week', 'month']

    predict_df = df.copy(deep=False)
    predict_df = predict_df[columns_kept]

    predict_df = predict_df.fillna(0)

    cat_cols = ['shipper', 'zone', 'week_number', 'day_of_week',
                'sender_state', 'recipient_state', 'month']

    float_cols = ['weight', 'distance', 'sender_pop', 'sender_pop_density', 'same_msa',
                  'sender_in_msa', 'rec_in_msa', 'sender_houses', 'recipient_pop', 'recipient_pop_density',
                  'recipient_houses']

    predict_df[cat_cols] = predict_df[cat_cols].astype('category')
    predict_df[float_cols] = predict_df[float_cols].astype('float64')

    print("One-hot-encoding features...")
    # Dummify dataframe
    predict_df = pd.get_dummies(predict_df)
    # Create empty dataframe in same shape as the one used in model, fill with 0s
    df_full = pd.DataFrame(columns=feature_names['feature_names_dummified'])
    # Execute a right join to align our test dataframe with full dataframe
    predict_df, df_full = predict_df.align(df_full, join='right', axis=1, fill_value=0)
    # Convert dataframe to numpy array for prediction
    X_test = predict_df.values
    print("Scaling data with saved scaler...")
    # Scale data with saved min-max scaler
    X_test = scaler.transform(X_test)
    utilities.print_elapsed_time(start_time)
    return df, X_test


def format_batch_results(pred, pred_proba, df):
    """Formats dataframe for output.

    Args:
        pred (int): Predicted ground delivery time window no. based on mode of predicted probability distribution.
        pred_proba (np array object): Predicted probability distribution across time windows.
            Shape of (no. time windows, no. of test shipments).
        df (pandas dataframe obj): Pandas dataframe

    Returns:
        Pandas dataframe object: Dataframe with scheduled time windows, predicted time windows, cumulative probability
            as columns.
    """
    print("Formatting batch results...")
    start_time = time.time()
    df_features = df.copy(deep=False)
    # Load dict that maps time window numbers to time windows
    windows_cmu = joblib.load(paths.windows_cmu)
    # Insert predicted time window into dataframe
    # windows_cmu maps time window number to corresponding time window
    df['scheduled_window'] = df.apply(lambda x: windows_cmu[x['scheduled_window_no']], axis=1)
    # Insert predicted probability distribution array object into each cell in column
    df['pred_ground_window_pdf'] = 0
    df['pred_ground_window_pdf'] = df['pred_ground_window_pdf'].astype(object)
    for i in range(len(df)):
        df.at[i, 'pred_ground_window'] = windows_cmu[pred[i]]
        df.at[i, 'pred_ground_window_pdf'] = pred_proba[i]
    # For each probability distribution array object, slice the array with scheduled window number
    # Sum of the sliced array is the predicted cumulative probability of shipment arriving before or in scheduled window
    df['prob_arrive_by_scheduled_window'] = df.apply(
        lambda x: np.sum(x['pred_ground_window_pdf'][:x['scheduled_window_no'] + 1]), axis=1)
    # Get cumulative probability for all time windows for each shipment
    col_to_format = ['prob_arrive_by_scheduled_window']
    for i in list(windows_cmu.keys()):
        col_name = "prob_arrive_by_window_" + str(i)
        df[col_name] = df.apply(lambda x: np.sum(x['pred_ground_window_pdf'][:i + 1]), axis=1)
        col_to_format.append(col_name)
    # Converts fraction into percentage for display
    df[col_to_format] = df[col_to_format].astype('float64')
    df[col_to_format] = df[col_to_format].values * 100
    df[col_to_format] = df[col_to_format].applymap("{0:.2f}%".format)
    # Drop columns that are not needed for prediction display
    df = df.drop(columns=['pred_ground_window_pdf', 'scheduled_window_no', 'week_number', 'day_of_week',
                          'month', 'distance', 'sender_in_msa', 'rec_in_msa', 'same_msa',
                          'sender_pop', 'sender_pop_density', 'sender_houses',
                          'sender_state', 'recipient_pop', 'recipient_pop_density',
                          'recipient_houses', 'recipient_state'])
    # Create df just for display
    df_display = df.copy(deep=False)
    col_to_format.pop(0)
    df_display = df_display.drop(columns=col_to_format)
    print(tabulate(df_display, headers='keys', showindex=False, floatfmt=".2f", tablefmt='psql'))
    # Save to xlsx format
    timestamp = utilities.get_timestamp()
    output_path = os.path.join(paths.output_delivery_prediction_dir, timestamp + "_predict_batch.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name='Predicted Window Probability')
        df_features.to_excel(writer, sheet_name='Features')
    print("\nResults saved to " + output_path)

    utilities.print_elapsed_time(start_time)
    return df


def predict_batch(df, model, scaler, feature_names):
    """Combines all preprocessing, prediction and formatting functions to return results.

    Args:
        df (pandas dataframe obj): Pandas dataframe from user CSV
        model (obj): Model loaded for prediction.
        scaler (obj): Scaler corresponding to selected model.
        feature_names (npz object): Dictionary of numpy arrays containing feature names.

    Returns:
        Pandas dataframe object: Dataframe with scheduled time windows, predicted time windows, cumulative probability
            as columns.
    """
    if validate_batch(df):
        print("Validation successful\n")
        df, X_test = preprocess_batch(df, feature_names, scaler)
        pred, pred_proba = predict_time_windows(X_test, model)
        df = format_batch_results(pred, pred_proba, df)
        return df
    else:
        print("\nPlease amend above errors before trying again.\n")
        from main import load_main_menu
        load_main_menu()


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    # Test get_MSA
    assert get_msa_details('15213', '15211') == (True, True, True)
    assert get_msa_details('00801', '15211') == (False, True, False)
    # Predict for one instance
    # predict_one_cost_savings("2019-07-09", "ups", 9, 91724, 15206)
    # Predict for batch
    df = pd.read_csv(paths.batch_sample_cmu, dtype=str)
    feature_names = np.load(paths.data_delivery_prediction_features_dir_cmu, allow_pickle=True)
    scaler = joblib.load(paths.scaler_cmu)
    model = joblib.load(paths.model_cmu)
    predict_batch(df, model, scaler, feature_names)
