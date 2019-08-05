"""
Data Extraction

This script contains the functions to extract data from 71lbs database and execute general cleaning
before passing the data on to other functions.

Extracted data is saved as compressed pickle files in extract_frac-<fraction>_<file_id>
where file_id is the YYYYMMDD-HHMM timestamp created when file storage is initialized, and
fraction is the fraction of data extracted.

This file should be imported as a module and contains the following functions that are used in main.py:
    * batch_query - Extracts a fraction of raw shipping records from the database.
    * store - Store the cleaned records as a compressed dataframe pickle object.

"""

import calendar
import os
import time
from datetime import datetime
import MySQLdb
import pandas as pd
from dateutil import relativedelta
from fuzzywuzzy import fuzz, process
import joblib
from tqdm import tqdm, trange
import credentials
import paths
import utilities


def query(db, start_date, end_date, frac):
    """Queries for and returns a sample of records that meet where clause criteria

    Args:
        db (obj): MySQL db object.
        start_date (str): Query start date in YYYY-MM-DD format.
        end_date (str): Query end date in YYYY-MM-DD format.
        frac (str): String representation of fraction of data to extract.

    Returns:
       Pandas dataframe object: Dataframe with extracted results.

    """
    # Initializes query based upon start and end year months of shipment_date
    fraction = utilities.string_to_decimal(frac)
    sql_query = """
    SELECT year_week, business_sid, UPPER(TRIM(industry)) AS industry, UPPER(TRIM(sub_industry)) AS sub_industry, shipper,
    TRIM(service_type_description) AS service_type, package_count, weight, 
    shipment_date, delivery_date, delivery_time, 
    freight_charges,freight_discount_amount,misc_charges,misc_discount_amount, 
    net_charge_amount, TRIM(zone) AS zone, UPPER(TRIM(sender_city)) AS sender_city, UPPER(TRIM(sender_state)) AS sender_state,
    LEFT(sender_zip,5) AS sender_zip, UPPER(TRIM(recipient_city)) AS recipient_city,
    UPPER(TRIM(recipient_state)) AS recipient_state, LEFT(recipient_zip,5) AS recipient_zip
    FROM libras.shipment_details 
    WHERE sender_country = 'US' 
    AND recipient_country = 'US' 
    AND delivery_date IS NOT NULL
    AND shipment_date >= STR_TO_DATE("{}", "%Y-%c-%e")
    AND shipment_date <= STR_TO_DATE("{}", "%Y-%c-%e")
    AND freight_charges > 0
    AND zone IS NOT NULL 
    AND zone !=''
    AND weight IS NOT NULL
    AND RAND() < {}
    """.format(start_date, end_date, fraction)
    records = pd.read_sql_query(sql_query, db)
    return records


def preprocess(records):
    """Preprocesses records to satisfy common cleansing requirements between benchmarking and delivery prediction solutions

    Args:
        records (pandas dataframe obj): Dataframe to be preprocessed.

    Returns:
        pandas dataframe obj: Dataframe with preprocessed records.

    """

    # Standardized shipping methods based primarily upon what is selectable through the FedEx API here:
    # https://www.fedex.com/ratefinder/home. 'Home Delivery' and 'Smartpost' are not selectable.
    fedex_methods = ['Same Day', 'First Overnight', 'Priority Overnight', 'Standard Overnight',
                     '2Day AM', '2Day', 'Express Saver', 'Ground', 'Home Delivery', 'Smartpost']

    # Standardized shipping methods based primarily on what is selectable through the API here:
    # https://wwwapps.ups.com/ctc/request?loc=en_US. 'Surepost' and 'Standard' are not selectable.
    ups_methods = ['Next Day Air Early', 'Next Day Air', 'Next Day Air Saver', '2nd Day Air A.M.',
                   '2nd Day Air', '3 Day Select', 'Ground', 'Surepost', 'Standard']

    # Standardized state names and codes of the 48 contiguous states based upon USPS standards found here:
    # https://www.ups.com/worldshiphelp/WS14/ENU/AppHelp/Codes/State_Province_Codes.htm
    state_codes = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'ID', 'IL', 'IN',
                   'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
                   'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                   'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

    # sets float dtypes and standardizes zones to single digits ranging between 2 and 8
    float_cols = ['freight_charges', 'freight_discount_amount', 'misc_charges', 'misc_discount_amount',
                  'net_charge_amount', 'zone']

    if len(records) > 0:
        records = records[records.zone.apply(lambda x: x.isnumeric())]
        records[float_cols] = records[float_cols].astype('float64')
        records.zone %= 10
        records['zone'] = records['zone'].astype('int')
        # Keep zones 2 to 8
        records = records[records['zone'].isin(range(2,9))]

        # strips leading and trailing whitespaces from all string values
        obj_columns = records.select_dtypes(include='object').columns
        for column in obj_columns:
            records[column] = records[column].str.strip()

        # converts a subset columns of dtype 'object' to dtype 'category' for memory conservation and later ml use
        cat_columns = ['industry', 'sub_industry', 'sender_state', 'recipient_state', 'zone']
        records[cat_columns] = records[cat_columns].astype('category')

        # creates std_weight (weight/package_count)
        records.insert(8, 'std_weight', records['weight'] / records['package_count'])

        # applies fuzzy macthing to each service type relative to the standardized list per carrier.
        service_type_fuzzy_match = []
        columns = ['shipper', 'service_type']
        for record in records[columns].itertuples():
            if record.shipper == 'fedex':
                service_type_fuzzy_match.append(
                    process.extractOne(record.service_type, fedex_methods, scorer=fuzz.partial_ratio))
            else:
                service_type_fuzzy_match.append(
                    process.extractOne(record.service_type, ups_methods, scorer=fuzz.partial_ratio))

        # adds the standardized service type and drops all records with service type scores less than 70
        records.insert(6, 'std_service_type', [method for method, score in service_type_fuzzy_match])
        records = records.assign(std_service_type_score=[score for method, score in service_type_fuzzy_match])
        records = records[records.std_service_type_score >= 70]

        # removes records with sender or recipient states residing outside of the 48 contiguous states
        records = records[(records.recipient_state.isin(state_codes + ['']))]
        records = records[(records.sender_state.isin(state_codes + ['']))]

        # drops unneeded columns
        records = records.drop(['std_service_type_score'], axis=1)

    return records


def batch_query(start_year_month, end_year_month, frac):
    """Extracts a fraction of raw shipping records from the database.

    Args:
        start_year_month (str): Start date in YYYY-MM format.
        end_year_month (str): End date in YYYY-MM format.\
        frac (str): String representation of fraction of data to extract.

    Returns:
        pandas dataframe obj: Dataframe with batch query records.
    """

    # Establishes connection to a MySQL db
    print(f"Connecting to {credentials.db}...")
    start_time = time.time()
    db = MySQLdb.connect(credentials.host, credentials.user, credentials.password, credentials.db)
    utilities.print_elapsed_time(start_time)

    # instantiates batch start and end dates as the int of the concatenated string of year + month
    print("Extracting and preprocessing records...")
    extraction_start_time = time.time()
    # Append day 1 to year and month to create datetime object. Day does not affect result
    start = datetime.strptime(start_year_month + "-1", "%Y-%m-%d").date()
    end = datetime.strptime(end_year_month + "-1", "%Y-%m-%d").date()
    records = pd.DataFrame()
    # Set warning for chained assignment to None.
    pd.options.mode.chained_assignment = None
    # Calculate number of months between start and end year/month
    delta = relativedelta.relativedelta(end, start)
    num_batches = delta.years * 12 + delta.months + 1
    month = start.month
    year = start.year
    # tqdm progress bar reference: https://github.com/tqdm/tqdm
    pbar = trange(num_batches)
    for i in pbar:
        pbar.set_description(f"Querying {year} {calendar.month_abbr[month]}")
        # First day of month in batch
        first_date_of_month = 1
        start_date = f"{year}-{month}-{first_date_of_month}"
        # Last day of month in batch
        last_date_of_month = calendar.monthrange(year, month)[1]
        end_date = f"{year}-{month}-{last_date_of_month}"
        # Query from first day to last day of given month
        results = query(db, start_date, end_date, frac)
        pbar.set_description(f"Preprocessing {len(results)} records for {year} {calendar.month_abbr[month]}")
        # Only preprocess if batch has records
        if len(results) > 0:
            records = records.append(preprocess(results), ignore_index=True)
        # If current month is 12, increment year by 1 and reset month to 1 for next batch
        if month == 12:
            month = 1
            year += 1
        # Else, increment month by 1
        else:
            month += 1
    print(f"{len(records)} records extracted and preprocessed")
    utilities.print_elapsed_time(extraction_start_time)
    return records


def create_filename(frac):
    """Creates file name from timestamp and fraction extracted.

    Args:
        frac (str): String representation of fraction of data to extract.

    Returns:
        str: File name for storing extracted data
    """
    file_id = utilities.get_timestamp()
    fraction = utilities.string_to_decimal(frac)
    filename = f"extract_frac-{int(fraction*100)}_{file_id}"
    return filename


def store(records, frac):
    """Stores results from each query into a compressed pickle file.

    Args:
        records (pandas dataframe obj): Dataframe after general preprocessing.
        frac (str): String representation of fraction of data to extract.

    Returns:
        str: Output file path

    """
    print("Saving records...")
    start_time = time.time()
    filename = create_filename(frac)
    output_path = os.path.join(paths.data_extracted_dir, filename+".pkl.z")
    joblib.dump(records, output_path)
    print(f"Data extracted and stored in {output_path}")
    utilities.print_elapsed_time(start_time)
    return output_path


if __name__ == '__main__':
    start, end, frac = ['2018-06', '2018-07', 0.01]
    records = batch_query(start, end, frac)
    print(records['zone'])
    print(records.iloc[0])
    output_path = store(records, frac)
