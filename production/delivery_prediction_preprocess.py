"""
Delivery Prediction Preprocessing Script

This script contains the functions to preprocess data before training a model.
Preprocessing involves removing unnecessary rows and columns, adding features based on
sender and recipient zipcodes, shipment dates, adding time windows as target variable Y,
splitting into training and test features and labels, and scaling continuous features e.g. weight.

Feature names and preprocessed data are saved as dictionaries feature_names_<model_id>_.npz
and datadict_<model_id>.npz respectively. Scaler is stored as a pkl object with .z compression in
scaler_<model_id>_.pkl.z. Model_id is the YYYYMMDD-HHMM timestamp created when
preprocessing is initialized.

This file should be imported as a module and contains the following
function that is used in main.py:
    * preprocess - combines all functions in this module to preprocess a dataframe.

"""
import os
import time
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import pgeocode
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from uszipcode import SearchEngine

import paths
import utilities


def set_model_id():
    """Gets timestamp as model_id to be used globally in this module.

    Returns:
        str: Current timestamp in YYYY-MM-DD format.

    """
    global model_id
    model_id = utilities.get_timestamp()
    return model_id


def report_df_stats(df):
    """Helper method to report on number of rows and columns in dataframe for tracking.

    Args:
        df (pandas dataframe obj): Pandas dataframe

    Returns:
        str: Number of rows and columns in dataframe.

    """
    rows = len(df)
    col = len(df.columns)
    return f"{rows} rows, {col} columns"


def remove_rows(df):
    """Removes unnecessary rows from dataframe.

    Args:
        df (pandas dataframe obj): Pandas dataframe

    Returns:
        pandas dataframe obj: Pandas dataframe with rows removed.
    """
    print("Removing unnecessary rows...")
    print(f"Starting with {report_df_stats(df)}")
    start_time = time.time()
    # Keep Ground and Home Delivery since both are day definite
    print("Removing non-ground service types since we are only predicting for ground deliveries...")
    services_kept = ["Ground", "Home Delivery"]
    df = df[df['std_service_type'].isin(services_kept)]
    # Keep rows with delivery time
    print("Removing rows without delivery time...")
    df = df.dropna(subset=['delivery_time'])
    # Remove rows that appear too often with abnormal delivery time = 23:59:59 and 00:12:00
    print("Removing rows with anomalous delivery time 23:59:59, 00.12.00...")
    df = df[~df['delivery_time'].isin(['23:59:59'])]
    df = df[~df['delivery_time'].isin(['00:12:00'])]
    # Trim empty spaces / Replace empty strings with NA / Remove NA
    print("Removing rows with malformed zipcodes...")
    df['recipient_zip'] = df['recipient_zip'].str.strip()
    df['recipient_zip'].replace('', np.nan, inplace=True)
    df['sender_zip'] = df['sender_zip'].str.strip()
    df['sender_zip'].replace('', np.nan, inplace=True)
    df.dropna(subset=['recipient_zip', 'sender_zip'], inplace=True)
    # Remove zipcodes with alphabets
    df = df[df['recipient_zip'].apply(lambda x: x.isnumeric())]
    df = df[df['sender_zip'].apply(lambda x: x.isnumeric())]
    ## Remove zipcodes if length != 5
    df = df[df['recipient_zip'].apply(lambda x: len(str(x)) == 5)]
    df = df[df['sender_zip'].apply(lambda x: len(str(x)) == 5)]
    print(f"Ending with {report_df_stats(df)}.")
    utilities.print_elapsed_time(start_time)
    return df


def add_MSA_features(df):
    """Adds MSA features for sender and recipient zipcodes to dataframe.

    sender_in_msa (bool): Whether sender zipcode is in a Metroplitan Statistical Area.
    rec_in_msa (bool): Whether recipient zipcode is in a Metroplitan Statistical Area.
    same_msa (bool): Whether sender zipcode and recipient have the same Metroplitan Statistical Area code.

    Args:
        df (pandas dataframe obj): Pandas dataframe that must contain sender_zip and recipient_zip columns

    Returns:
        pandas dataframe obj: Pandas dataframe with new columns same_msa, rec_in_msa, same_msa.
    """
    df = df.reset_index(drop=True)
    print("Adding MSA details...")
    zipcode_to_msa_df = pd.read_csv(paths.data_delivery_prediction_zip_to_msa_cmu, dtype=object)
    # If MSA Name contains 'MSA', zipcode is in a MSA i.e. bool=True
    zipcode_to_msa_df['MSA'] = zipcode_to_msa_df['MSA Name'].str.contains(r'MSA')
    # Only keep rows where zipcodes are MSAs
    zipcode_to_msa_df = zipcode_to_msa_df[zipcode_to_msa_df['MSA'].isin([True])]
    # Create pd series for all zipcodes in our df
    s = pd.Series(list(df['sender_zip'].values))
    r = pd.Series(list(df['recipient_zip'].values))
    # Add columns to df for all zipcodes that are found in zipcode column of zipcode_to_msa_df
    df['sender_in_msa'] = s.isin(zipcode_to_msa_df['ZIP CODE'])
    df['rec_in_msa'] = r.isin(zipcode_to_msa_df['ZIP CODE'])
    zip_to_msa_dict = zipcode_to_msa_df.set_index('ZIP CODE')['MSA No.'].to_dict()
    same_msa_list = []
    for row in df.itertuples():
        try:
            if zip_to_msa_dict[row.sender_zip] == zip_to_msa_dict[row.recipient_zip]:
                same_msa_list.append(True)
            else:
                same_msa_list.append(False)
        except KeyError:
            same_msa_list.append(False)
    df['same_msa'] = same_msa_list
    return df

def get_distance(zipcode1, zipcode2):
    """Gets geodesic distance between 2 given zipcodes

    Reference: https://pgeocode.readthedocs.io/en/latest/overview.html

    Args:
        zipcode1 (str): sender 5-digit zipcode
        zipcode2 (str): recipient 5-digit zipcode

    Returns:
        float: distance between zipcodes in miles
    """
    dist = pgeocode.GeoDistance('us')
    distance = dist.query_postal_code(zipcode1, zipcode2)
    if distance is not None:
        return distance
    else:
        return 0


def get_zip_details(zipcode1, search):
    """Get population, population density, no. housing units and state for given zipcode.

    Reference: https://pgeocode.readthedocs.io/en/latest/overview.html

    Args:
        zipcode1 (str): target 5-digit zipcode
        search (obj): uszipcode search object

    Returns:
        pop (int): population
        pop_density (int): population density
        housing_units (int): number of housing units
    """

    zipcode = search.by_zipcode(zipcode1)
    pop = zipcode.population
    pop_density = zipcode.population_density
    housing_units = zipcode.housing_units
    # Returns 0 if not found. If NaN, program will encounter error
    # during transformation.
    return pop or 0, pop_density or 0, housing_units or 0


def add_zip_details(df):
    """Adds zip details retrieved with get_zip_details() to Dataframe

    Args:
        df (pandas dataframe obj): Pandas dataframe that must contain sender_zip and recipient_zip columns

    Returns:
        pandas dataframe obj: Pandas dataframe with new columns sender_pop, sender_pop_density, sender_houses,
            recipient_pop, recipient_pop_density, recipient_houses
    """
    print(f"Adding zipcode details, {len(df)} iterations expected...")
    search = SearchEngine()
    pop_list_s, pop_density_list_s, houses_list_s = [], [], []
    pop_list_r, pop_density_list_r, houses_list_r = [], [], []
    for row in tqdm(df.itertuples()):
        pop_s, pop_density_s, houses_s = get_zip_details(row.sender_zip, search)
        pop_list_s.append(pop_s)
        pop_density_list_s.append(pop_density_s)
        houses_list_s.append(houses_s)
        pop_r, pop_density_r, houses_r = get_zip_details(row.recipient_zip, search)
        pop_list_r.append(pop_r)
        pop_density_list_r.append(pop_density_r)
        houses_list_r.append(houses_r)
    df['sender_pop'], df['sender_pop_density'], df['sender_houses'] = pop_list_s, pop_density_list_s, houses_list_s
    df['recipient_pop'], df['recipient_pop_density'], df['recipient_houses'] = pop_list_r, pop_density_list_r, houses_list_r
    return df


def add_features(df):
    """Combines other functions to add features to dataframe.

    Args:
        df (pandas dataframe obj): Pandas dataframe that must contain sender_zip, recipient_zip,
            shipment_date, weight, package_count columns.

    Returns:
        pandas dataframe obj: Pandas dataframe with new columns sender_pop, sender_pop_density, sender_houses,
            recipient_pop, recipient_pop_density, recipient_houses, week_number, day_of_week, month, std_weight,
            distance, sender_in_msa, rec_in_msa, same_msa.
    """
    print("Adding features...")
    print(f"Starting with {report_df_stats(df)}")
    start_time = time.time()
    # Create std_weight (weight/package count)
    df['std_weight'] = df['weight'] / df['package_count']
    # Add date time features based on shipment date
    print("Adding datetime features based on shipment date...")
    df['week_number'] = df['shipment_date'].dt.week
    df['day_of_week'] = df['shipment_date'].dt.dayofweek
    df['month'] = df['shipment_date'].dt.month
    # Add distance between sender and recipient zips
    print("Adding distance...")
    df['distance'] = get_distance(df['sender_zip'].values, df['recipient_zip'].values)
    # Add sender_in_MSA, rec_in_MSA, same_MSA bools
    df = add_MSA_features(df)
    # Add population, population density, no. housing units
    add_zip_details(df)
    print(f"Ending with {report_df_stats(df)}.")
    utilities.print_elapsed_time(start_time)
    return df


def add_time_windows(df):
    """Adds time windows as target variable, based on delivery date and time.

    Calculates number of business days (excl weekends, shipment date, federal holidays) for delivery.
    Adds delivery time to number of business days to get time-in-transit.
    Creates window thresholds based on discussion with Jose.
    Lastly assign time windows to shipments based on time-in-transit.

    Args:
        df (pandas dataframe obj): Pandas dataframe that must contain shipment_date, delivery_date, delivery_time.

    Returns:
        pandas dataframe obj: Pandas dataframe with new columns days_in_transit, days_taken_float (time-in-transit),
            Y (target variable i.e. time window)

    """
    print("Adding time windows i.e. target variable...")
    print(f"Starting with {report_df_stats(df)}")
    start_time = time.time()
    # Calculate days in transit (exclude shipment date, holidays, weekends)
    start_date = df['shipment_date'].min()
    end_date = df['shipment_date'].max()
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start_date, end_date).date.tolist()
    shipment_dates = [d.date() for d in df['shipment_date']]
    delivery_dates = [d.date() for d in df['delivery_date']]
    # -1 because we will add transit time
    df['days_in_transit'] = np.busday_count(shipment_dates, delivery_dates,
                                                      holidays=holidays) - 1
    # Convert days in transit/delivery time to days taken (with decimals))
    # e.g. if parcel reaches at 12.00pm on 2nd business day, days taken is 1.5
    delivery_percentage_of_day = [(timedelta.total_seconds(d) / timedelta(days=1).total_seconds()) for d in
                                  df['delivery_time']]
    df['days_taken_float'] = df['days_in_transit'] + delivery_percentage_of_day
    # Keep rows from -1 to 5 days in transit. The rest are rare occurrences.
    max_days_to_keep = 5
    df = df[df['days_in_transit'].isin(np.arange(-1,max_days_to_keep))]

    # Assign time windows
    time_window_thresholds = create_time_window_thresholds()
    tqdm.pandas(desc="Assign time window")
    df['Y'] = df.progress_apply(lambda x: assign_time_window(x['days_taken_float'], time_window_thresholds), axis=1)

    print(f"Ending with {report_df_stats(df)}.")
    utilities.print_elapsed_time(start_time)
    return df


def create_time_window_thresholds():
    """Create 12 time window thresholds according to percentage of days.
    
    Window 0: 0D,
    Window 1: 1D 00:00 - 07:59,
    Window 2: 1D 08:00 - 10:29,
    Window 3: 1D 10:30 - 14:59, 
    Window 4: 1D 15:00 - 23:59,
    Window 5: 2D 00:00 - 10:29,
    Window 6: 2D 10:30 - 16:29,
    Window 7: 2D 16:30 - 23:59,
    Window 8: 3D 00:00 - 16:29,
    Window 9: 3D 16:30 - 23:59,
    Window 10: 4D 00:00 - 23:59,
    Window 11: 5D 00:00 - 23:59
 
    Returns:
        List of time window thresholds according to percentage of days.
    """
    # Create time windows
    percentage_of_day_list = []
    # 8.00am
    eight_am = timedelta(hours=8, minutes=0).total_seconds() / timedelta(days=1).total_seconds()
    percentage_of_day_list.append(eight_am)
    # 10.30am
    ten_thirty_am = timedelta(hours=10, minutes=30).total_seconds() / timedelta(days=1).total_seconds()
    percentage_of_day_list.append(ten_thirty_am)
    # 3.00pm
    three_pm = timedelta(hours=15, minutes=0).total_seconds() / timedelta(days=1).total_seconds()
    percentage_of_day_list.append(three_pm)
    # 4.30pm
    four_thirty_pm = timedelta(hours=16, minutes=30).total_seconds() / timedelta(days=1).total_seconds()
    percentage_of_day_list.append(four_thirty_pm)

    print(f"8.00am: {eight_am} day, \
    10.30am: {ten_thirty_am} day, \
    3.00pm: {three_pm} day, \
    4.30pm: {four_thirty_pm} day")

    # Create time window thresholds
    time_window_thresholds = [eight_am, ten_thirty_am, three_pm, 1,
                              1 + ten_thirty_am, 1 + four_thirty_pm, 2,
                              2 + four_thirty_pm, 3,
                              4, 5]
    return time_window_thresholds


def assign_time_window(time_in_transit, time_window_thresholds):
    """Helper function to assign time windows based on time in transit.
    
    Args:
        time_in_transit: Number of business days + delivery time in decimals
        time_window_thresholds: List of time window thresholds according to percentage of days.

    Returns:
        int: Time window number that time_in_transit belongs to.
    """
    # If 0 business days, assign to window 0
    if time_in_transit <= 0: return 0
    # If >0 business days, assign to respective time window
    for upper_bound in time_window_thresholds:
        if time_in_transit <= upper_bound:
            # Use index of thresholds to assign time windows
            return time_window_thresholds.index(upper_bound)+1


def remove_columns(df):
    """Removes columns not needed for prediction.
    
    Args:
        df (pandas dataframe obj): Pandas dataframe 

    Returns:
        pandas dataframe obj: Pandas dataframe with only shipper, std_weight, zone, 
            sender_state, recipient_state, distance, sender_pop, sender_pop_density, sender_houses,
            recipient_pop, recipient_pop_density, recipient_houses, same_msa,
            sender_in_msa, rec_in_msa, week_number, day_of_week, month, Y columns
            
    """
    print("Removing unnecessary columns...")
    print(f"Starting with {report_df_stats(df)}\nRemoving columns...")
    start_time = time.time()
    columns_kept = ['shipper','std_weight','zone',
                    'sender_state','recipient_state', 'distance',
                    'sender_pop', 'sender_pop_density', 'sender_houses',
                    'recipient_pop', 'recipient_pop_density', 'recipient_houses',
                    'same_msa', 'sender_in_msa', 'rec_in_msa',
                    'week_number','day_of_week','month','Y']
    df = df[columns_kept]
    nan_rows = df[df.isnull().T.any().T]
    nan_rows.to_csv('null.csv')
    print(f"Ending with {report_df_stats(df)}.")
    utilities.print_elapsed_time(start_time)
    return df


def one_hot_encode(df):
    """Categorize and apply one-hot-encoding to all categorical columns.

    Args:
        df (pandas dataframe obj): Pandas dataframe with only columns used for prediction

    Returns:
        X (pandas dataframe obj): Pandas dataframe with dummified categorical features and other features.
        y (pandas dataframe obj): Pandas dataframe with target variable i.e. time window number.
    """
    print("Categorizing columns...")
    print(f"Starting with {report_df_stats(df)}")
    start_time = time.time()
    cat_cols = ['shipper', 'zone',
                'sender_state', 'recipient_state',
                'week_number', 'day_of_week', 'month']

    float_cols = ['std_weight', 'distance',
                  'sender_pop', 'sender_pop_density', 'sender_houses',
                  'recipient_pop', 'recipient_pop_density', 'recipient_houses']

    df[cat_cols] = df[cat_cols].astype('category')
    df[float_cols] = df[float_cols].astype('float64')

    print("Number of nulls in each column:")
    print(df.isnull().sum())
    print("\nReplacing nulls with mean/mode...")
    df[float_cols] = df[float_cols].fillna(df[float_cols].mean())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode())
    print("Number of nulls in each column after replacement:")
    print(df.isnull().sum(), "\n")

    print("Applying one hot encoding on categorical columns...")
    y = df.Y
    df = df.drop(columns=['Y'])
    # Record feature names for current model
    feature_names = {'feature_names': df.columns.values}
    X = pd.get_dummies(df)
    feature_names['feature_names_dummified'] = X.columns.values
    print("\nFeature names:", feature_names['feature_names_dummified'])
    # Save feature names for use in prediction
    features_path = os.path.join(paths.data_delivery_prediction_features_dir, "feature_names_" + model_id + ".npz")
    np.savez(features_path,
            feature_names=feature_names['feature_names'],
            feature_names_dummified=feature_names['feature_names_dummified'])
    print(f"Feature names stored in {features_path}\n")
    print(f"Ending with {report_df_stats(X)}.")
    utilities.print_elapsed_time(start_time)
    return X, y


def split_scale_data(X, y):
    '''Split into training and test datasets, and scale data

    Args:
        X (pandas dataframe obj): Pandas dataframe with dummified categorical features and other features.
        y (pandas dataframe obj): Pandas dataframe with target variable i.e. time window number.

    Returns:
        Dictionary of numpy arrays:
            X_train: the matrix of training data
            y_train: the array of training labels
            X_test: the matrix of testing data
            y_test: the array of testing labels
    '''
    # Split data
    print("Splitting and scaling data...")
    print("Splitting data...")
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=71)
    # Scale the variables
    print("Scaling data...")
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Save scaler for use in prediction
    print("Saving scaler...")
    scaler_path = os.path.join(paths.model_scaler_dir, "scaler_" + model_id + ".pkl.z")
    joblib.dump(scaler, scaler_path)
    print("Scaler saved in", scaler_path)
    # return training and testing data
    datadict = {'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test}
    # Save data dictionary file
    print("\nSaving data dictionary...")
    datadict_path = os.path.join(paths.data_delivery_prediction_datadict_dir, "datadict_" + model_id + ".npz")
    np.savez(datadict_path,
            X_train=datadict['X_train'],
            y_train=datadict['y_train'],
            X_test=datadict['X_test'],
            y_test=datadict['y_test'])
    print("Data dictionary saved in", datadict_path)
    # Print sizes
    print("\nTraining and testing dataset sizes")
    print("X_train", X_train.shape, "y_train", y_train.shape)
    print("X_test", X_test.shape, "y_test", y_test.shape)
    utilities.print_elapsed_time(start_time)
    return datadict


def preprocess(extracted_data):
    """Combines all preprocessing functions. Invoked by main.

    Args:
        extracted_data (pandas df obj): Pandas dataframe containing data extracted from 71lbs database

    Returns:
        datadict (dict): Dictionary of numpy arrays containing preprocessed train and test data.
        model_id (str): Timestamp used to identify model, scaler and feature names files

    """
    # Set warning for chained assignment to None.
    print("Preprocessing data before training...\n")
    pd.options.mode.chained_assignment = None
    model_id = set_model_id()
    df = remove_rows(extracted_data)
    df = add_features(df)
    df = add_time_windows(df)
    df = remove_columns(df)
    X, y = one_hot_encode(df)
    datadict = split_scale_data(X, y)
    return datadict, model_id

if __name__ == '__main__':
    df = joblib.load(os.path.join(paths.extracted_data_sample_cmu))
    preprocess(df)

