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
import credentials
import paths


def create_id():
    now = datetime.now()
    model_id = now.strftime("%Y%m%d-%H%M")
    return model_id


def print_elapsed_time(start_time):
    elapsed_time_secs = time.time() - start_time
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
    print(msg)


def get_distance(zipcode1, zipcode2):
    dist = pgeocode.GeoDistance('us')
    return dist.query_postal_code(zipcode1, zipcode2)


def get_zip_details(zipcode1, search):
    zipcode = search.by_zipcode(zipcode1)
    pop = zipcode.population
    pop_density = zipcode.population_density
    housing_units = zipcode.housing_units
    state = zipcode.state
    # Return 0 if not found. nans will encounter error later.
    return pop or 0, pop_density or 0, housing_units or 0, state or 0


def get_date_details(shipment_date):
    shipment_date_parsed = pd.Series(datetime.strptime(shipment_date, '%Y-%m-%d'))
    week_number = shipment_date_parsed.dt.week.values[0]
    day_of_week = shipment_date_parsed.dt.dayofweek.values[0]
    month = shipment_date_parsed.dt.month.values[0]
    return week_number, day_of_week, month


def get_msa_details(sender_zip, recipient_zip):
    zipcode_to_msa_df = pd.read_csv(paths.data_delivery_prediction_msa_dir_cmu, dtype=object)
    zipcode_to_msa_df.columns = ['zipcode', 'state', 'msa_num', 'county_num', 'msa_name']
    zip_msa_num_dict = zipcode_to_msa_df.set_index('zipcode')['msa_num'].to_dict()
    zip_msa_name_dict = zipcode_to_msa_df.set_index('zipcode')['msa_name'].to_dict()
    # Make sure zips are strings
    sender_zip = str(sender_zip)
    recipient_zip = str(recipient_zip)
    # Finding sender zip MSA number and adding boolean for if in MSA
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
    # Finding recipient zip MSA number and adding boolean for if in MSA
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
    # Checking to see if sender and recipient are in same MSA and filling list
    if sender_msa_num == recipient_msa_num:
        same_msa = 1
    else:
        same_msa = 0
    return sender_in_msa, rec_in_msa, same_msa


def get_shippo_details(shipper, weight, sender_zip, sender_state, recipient_zip, recipient_state):
    shipper = str.lower(shipper)
    shippo.api_key = credentials.shippo_test_key
    # The complete reference for the address object is available here: https://goshippo.com/docs/reference#addresses
    address_from = {"state": sender_state, "zip": sender_zip, "country": "US"}
    address_to = {"state": recipient_state, "zip": recipient_zip, "country": "US"}
    parcel = {
        "length": "5", "width": "5", "height": "5", "distance_unit": "in", # dummy dimensions
        "weight": weight, "mass_unit": "lb"
    }
    shipment = shippo.Shipment.create(
        address_from=address_from,
        address_to=address_to,
        parcels=[parcel],
        asynchronous=False
    )
    shippo_id = shippo.Shipment.retrieve(shipment.object_id)['object_id']
    rates = shippo.Shipment.get_rates(shippo_id)
    # Get zone and insert retrieved rates in dict
    results = {}
    zone = 5  # default
    for i in rates['results']:
        results[i['servicelevel']['token']] = i['amount']
        if i['servicelevel']['token'] == "fedex_ground":
            zone = i['zone']
    # If dictionary is empty, means no results were returned
    if not bool(results):
        print("No results from Shippo. Please try again.")
        return None, None
    else:
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
        headers = ['service','ship_cost', 'ground_cost', 'cost_saving', 'scheduled_window_no', 'scheduled_window']
        if shipper == 'ups':
            chosen_dict = ups_services_dict
        elif shipper == 'fedex':
            chosen_dict = fedex_services_dict
        # Create lists
        services = []
        ship_costs = []
        ground_costs = []
        cost_savings = []
        scheduled_windows = []
        scheduled_window_nos = []
        try:
            if shipper == 'ups':
                ground_cost_ = float(results['ups_ground'])
            elif shipper == 'fedex':
                ground_cost_ = float(results['fedex_ground'])
        except KeyError:
            print(shipper, "does not send to the zipcode provided. Please try again.")
        # Load window to window time dict
        windows_cmu = joblib.load(paths.windows_cmu)
        # Append values to list
        for i in chosen_dict.keys():
            if i in results.keys():
                services.append(i)
                ship_costs.append(results[i])
                ground_costs.append(ground_cost_)
                cost_savings.append(float(results[i]) - ground_cost_)
                scheduled_window_no = chosen_dict[i]
                scheduled_window_nos.append(scheduled_window_no)
                scheduled_windows.append(windows_cmu[scheduled_window_no])
        # Create list of tuples for dataframe
        df = pd.DataFrame(list(zip(services, ship_costs, ground_costs, cost_savings, scheduled_window_nos, scheduled_windows)), columns=headers)
        print(df)
        return df, zone


def preprocess_one(shipment_date, shipper, std_weight, sender_zip, recipient_zip, scaler):
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
    # Populate dataframe
    # Create empty dataframe with correct columns
    feature_names = np.load(paths.data_delivery_prediction_features_dir_cmu, allow_pickle=True)
    df = pd.DataFrame(columns=feature_names['feature_names'])
    # Add row into df
    df.loc[0] = [shipper, std_weight, zone,
                 sender_state, recipient_state, distance, sender_pop, sender_pop_density,
                 sender_houses, recipient_pop, recipient_pop_density, recipient_houses, same_msa,
                 sender_in_msa, rec_in_msa, week_number, day_of_week, month]
    # print(df.loc[0])
    #############################################
    #     # For MINMAX SCALED DISPLAY ONLY, to remove in production
    display_df = df.copy(deep=False)
    display_df = pd.concat([display_df] * 2, ignore_index=True)
    #############################################
    # Define categorical and float columns
    cat_cols = ['shipper', 'zone', 'week_number', 'day_of_week',
                'sender_state', 'recipient_state', 'month']
    float_cols = ['std_weight', 'distance', 'sender_pop', 'sender_pop_density', 'sender_houses', 'recipient_pop', 'recipient_pop_density',
                  'recipient_houses']
    df[cat_cols] = df[cat_cols].astype('category')
    df[float_cols] = df[float_cols].astype('float64')
    # Dummify dataframe
    df = pd.get_dummies(df)
    # Create empty dataframe in same shape as the one used in model, fill with 0s
    df_full = pd.DataFrame(columns=feature_names['feature_names_dummified'])
    # Execute a right join to align our test dataframe with full dataframe
    df, df_full = df.align(df_full, join='right', axis=1, fill_value=0)
    # Convert dataframe to numpy array for prediction
    test = df.loc[0].values
    # Scale data with saved min-max scaler
    test = test.reshape(1, -1)
    scaler = joblib.load(scaler)
    test = scaler.transform(test)
    #############################################
    # For MINMAX SCALED DISPLAY ONLY, to remove in production
    np.set_printoptions(precision=5, suppress=True)
    test_display = np.array(test[0], copy=True)
    test_display = np.around(test_display, 5)
    display_df.at[1, 'std_weight'] = test_display[0]
    display_df.at[1, 'distance'] = test_display[1]
    display_df.at[1, 'sender_pop'] = test_display[2]
    display_df.at[1, 'sender_pop_density'] = test_display[3]
    display_df.at[1, 'sender_houses'] = test_display[4]
    display_df.at[1, 'recipient_pop'] = test_display[5]
    display_df.at[1, 'recipient_pop_density'] = test_display[6]
    display_df.at[1, 'recipient_houses'] = test_display[7]
    display_df_T = display_df.T
    display_df_T['features'] = display_df.columns.values
    # Reverse order of columns
    display_df_T = display_df_T[['features', 0, 1]]
    print(tabulate(display_df_T, headers=['features','raw','scaled'], showindex=False, floatfmt=".5f", tablefmt='psql'))
    ##############################################
    print_elapsed_time(start_time)
    print(" ")
    return test, rates_df


def predict_time_windows(test, model):
    start_time = time.time()
    print("Loading trained model and predicting time windows...")
    # disabled for demo
    # model = joblib.load(model)
    pred = model.predict(test)
    pred_proba = model.predict_proba(test)
    print_elapsed_time(start_time)
    print(" ")
    return pred, pred_proba


def predict_cost_savings(pred, pred_proba, rates_df):
    start_time = time.time()
    print("Calculating cost savings and probabilities...")
    # Load window to window time dict
    windows_cmu = joblib.load(paths.windows_cmu)
    # Insert predicted time window into df
    rates_df['pred_ground_window'] = windows_cmu[pred]
    # Insert predicted array into df
    rates_df['pred_ground_window_pdf'] = 0
    rates_df['pred_ground_window_pdf'] = rates_df['pred_ground_window_pdf'].astype(object)
    for i in range(len(rates_df)):
        rates_df.at[i, 'pred_ground_window_pdf'] = pred_proba
    # For each row, calculate ground CDF up till scheduled window
    rates_df['pred_probability'] = rates_df.apply(lambda x: np.sum(x['pred_ground_window_pdf'][:x['scheduled_window_no'] + 1]), axis=1)
    rates_df['pred_probability'] = pd.Series(["{0:.2f}%".format(val * 100) for val in rates_df['pred_probability']], index=rates_df.index)
    rates_df = rates_df.drop(columns=['pred_ground_window_pdf', 'scheduled_window_no'])
    print_elapsed_time(start_time)
    print(" ")
    return rates_df


def predict_one_cost_savings(shipment_date, shipper, weight, sender_zip, recipient_zip,
                             scaler=paths.scaler_cmu, model=paths.model_cmu):
    model_id = create_id()
    preprocessed_input, rates_df = preprocess_one(shipment_date, shipper, weight, sender_zip, recipient_zip, scaler)
    pred, pred_proba = predict_time_windows(preprocessed_input, model)
    cost_savings_df = predict_cost_savings(pred[0], pred_proba[0], rates_df)
    # Save output to CSV
    output_path = os.path.join(paths.output_delivery_prediction_dir, model_id)
    cost_savings_df.to_csv(output_path+"_predict_one.csv")
    # Print output
    print(tabulate(cost_savings_df, headers='keys', showindex=False, floatfmt=".2f", tablefmt='psql',
                   colalign=['left', 'center', 'center', 'center', 'center', 'center', 'center']))
    print("Results saved to " + output_path + "_predict_one.csv")
    return cost_savings_df


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    # Predict for one instance
    predict_one_cost_savings("2019-07-09", "ups", 9, 91724, 15206)
    # Todo predict cost savings for batch
    # Todo predict time windows for batch


# def preprocess_batch(csv_file):
#     # Read CSV file
#     csv = pd.read_csv(csv_file, dtype={'recipient_zip': str, 'sender_zip': str})
#     # Create empty dataframe with correct columns and length of CSV file
#     feature_names = np.load('data/feature_names.npz')
#     # Get datetime features
#     csv['shipment_date'] = pd.to_datetime(csv['shipment_date'])
#     csv['week_number'] = csv['shipment_date'].dt.week
#     csv['day_of_week'] = csv['shipment_date'].dt.dayofweek
#     csv['month'] = csv['shipment_date'].dt.month
#     # Get sender_in_msa and recipient_in_MSA and same_msa booleans
#     zipcode_to_msa_df = pd.read_csv('data/zip_to_MSA_numbers.csv', dtype=object)
#     zipcode_to_msa_df.columns = ['zipcode', 'state', 'msa_num', 'county_num', 'msa_name']
#     zip_msa_num_dict = zipcode_to_msa_df.set_index('zipcode')['msa_num'].to_dict()
#     zip_msa_name_dict = zipcode_to_msa_df.set_index('zipcode')['msa_name'].to_dict()
#     # Lists to be filled and then converted to dataframe column features
#     sender_msa_num = []
#     sender_in_msa = []
#     recipient_msa_num = []
#     recipient_in_MSA = []
#     send_rec_same_msa = []
#     # For debugging purposes (find zipcodes that don't show up in dictionary)
#     zips_not_in_dict = {}
#
#     for row in csv.itertuples():
#         if row.sender_zip in zip_msa_num_dict:
#             sender_msa_num.append(zip_msa_num_dict[row.sender_zip])
#             msa_name = zip_msa_name_dict[row.sender_zip]
#             if 'MSA' in msa_name:
#                 sender_in_msa.append(1)
#             else:
#                 sender_in_msa.append(0)
#         else:
#             sender_in_msa.append(0)
#             sender_msa_num.append(0)
#             if row.sender_zip not in zips_not_in_dict:
#                 zips_not_in_dict[row.sender_zip] = 1
#             else:
#                 zips_not_in_dict[row.sender_zip] += 1
#         if row.recipient_zip in zip_msa_num_dict:
#             recipient_msa_num.append(zip_msa_num_dict[row.recipient_zip])
#             msa_name = zip_msa_name_dict[row.recipient_zip]
#             if 'MSA' in msa_name:
#                 recipient_in_MSA.append(1)
#             else:
#                 recipient_in_MSA.append(0)
#         else:
#             recipient_msa_num.append(0)
#             recipient_in_MSA.append(0)
#             if row.recipient_zip not in zips_not_in_dict:
#                 zips_not_in_dict[row.recipient_zip] = 1
#             else:
#                 zips_not_in_dict[row.recipient_zip] += 1
#     # Checking to see if sender and recipient are in same MSA and filling list
#     for s, r in zip(sender_msa_num, recipient_msa_num):
#         if s == r:
#             send_rec_same_msa.append(1)
#         else:
#             send_rec_same_msa.append(0)
#     # Creating columns and adding to dataframe
#     csv['same_msa'] = pd.Series(send_rec_same_msa)
#     csv['sender_in_msa'] = pd.Series(sender_in_msa)
#     csv['rec_in_msa'] = pd.Series(recipient_in_MSA)
#     # Get distance
#     csv['distance'] = get_distance(csv['sender_zip'].values, csv['recipient_zip'].values)
#     # Get population, density, no. houses, state code for recipient and sender
#     csv['sender_pop'], csv['sender_pop_density'], csv['sender_houses'], csv['sender_state'] = \
#         zip(*csv.apply(lambda x: get_zip_details(x['sender_zip']), axis=1))
#
#     csv['recipient_pop'], csv['recipient_pop_density'], csv['recipient_houses'], csv['recipient_state'] = \
#         zip(*csv.apply(lambda x: get_zip_details(x['recipient_zip']), axis=1))
#     # Decide on columns to keep
#     columns_kept = ['shipper', 'std_weight', 'zone',
#                     'sender_state', 'recipient_state', 'distance', 'sender_pop', 'sender_pop_density',
#                     'sender_houses', 'recipient_pop', 'recipient_pop_density', 'recipient_houses', 'same_msa',
#                     'sender_in_msa', 'rec_in_msa', 'week_number', 'day_of_week', 'month']
#
#     df = csv[columns_kept]
#
#     # Convert all nans to zeros
#     df = df.fillna(0)
#
#     # Define categorical and float columns
#     cat_cols = ['shipper', 'zone', 'week_number', 'day_of_week',
#                 'sender_state', 'recipient_state', 'month']
#
#     float_cols = ['std_weight', 'distance', 'sender_pop', 'sender_pop_density', 'same_msa',
#                   'sender_in_msa', 'rec_in_msa', 'sender_houses', 'recipient_pop', 'recipient_pop_density',
#                   'recipient_houses']
#
#     df[cat_cols] = df[cat_cols].astype('category')
#     df[float_cols] = df[float_cols].astype('float64')
#     # Dummify dataframe
#     df = pd.get_dummies(df)
#     # Create empty dataframe in same shape as the one used in model, fill with 0s
#     df_full = pd.DataFrame(columns=feature_names['feature_names_dummified'])
#     # Execute a right join to align our test dataframe with full dataframe
#     df, df_full = df.align(df_full, join='right', axis=1, fill_value=0)
#     # Convert dataframe to numpy array for prediction
#     X_test = df.values
#     # Scale data with saved min-max scaler
#     scaler = joblib.load(paths.scaler_cmu)
#     X_test = scaler.transform(X_test)
#     return X_test
