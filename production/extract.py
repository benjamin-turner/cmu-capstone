import os
import math

import time
from datetime import datetime
from datetime import timedelta
from datetime import date

import pandas as pd
import MySQLdb
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import paths, utilities, credentials


def query(db, start_year_week, end_year_week, frac=1):
    """
    Queries for and returns a sample of records that meet where clause criteria
    """
    # initializes query based upon start and end dates of shipment_date

    fraction = utilities.string_to_decimal(frac)

    sql_query = """
    select * from 
    (select * from 
    (select * from 
    (select year_week, business_sid, upper(trim(industry)) as industry, upper(trim(sub_industry)) as sub_industry,shipper,
    trim(service_type_description) as service_type,package_count, weight,shipment_date,delivery_date, delivery_time, 
    freight_charges,freight_discount_amount,misc_charges,misc_discount_amount, 
    net_charge_amount, zone, upper(trim(sender_city)) as sender_city, upper(trim(sender_state)) as sender_state,
    left(sender_zip,5) as sender_zip, upper(trim(recipient_city)) as recipient_city,
    upper(trim(recipient_state)) as recipient_state, left(recipient_zip,5) as recipient_zip 
    from libras.shipment_details 
    where sender_country = 'US' and recipient_country = 'US' and delivery_date is not null
    and year_week >= {} and year_week < {} and rand() < {}
    ) t1 
    where t1.shipment_date is not null) t2 
    where t2.freight_charges > 0) t3 
    where t3.zone is not null or trim(zone)!='' 
    """.format(start_year_week, end_year_week, fraction)

    # queries database and returns a sample of results
    # records = pd.read_sql_query(sql_query, db).sample(frac=fraction, replace=False)
    records = pd.read_sql_query(sql_query, db)
    return records


def preprocess(records):
    """
    Preprocesses records to satisfy common cleansing requirements between benchmarking and delivery prediction solutions
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
        records = records[(records.zone >= 2) & (records.zone <= 8)]
        records.zone = records.zone.__str__()

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



def batch_query(start, end, frac):
    """
    Extracts a fraction of raw shipping records from the database.

    :param start: datetime
        The earliest ship date of parcels
    :param end: datetime
    :param frac: float
        Ranges from 0 to 1
    :return: pandas dataframe
    """

    # Establishes connection to a MySQL db
    print(f"Connecting to {credentials.db}...")
    start_time = time.time()
    db = MySQLdb.connect(credentials.host, credentials.user, credentials.password, credentials.db)
    utilities.print_elapsed_time(start_time)

    # instantiates batch start and end dates as the int of the concatenated string of year + month
    print("Extracting and preprocessing records...")
    extraction_start_time = time.time()
    start = datetime.strptime(start, "%Y-%m-%d").date()
    end = datetime.strptime(end, "%Y-%m-%d").date()
    now = datetime.now().date()
    now_week = math.ceil(((now - date(year=now.year, month=1, day=1)).days / 365) * 52)
    now_year_week = int(now.year.__str__() + now_week.__str__().zfill(2))
    batch_start = start
    batch_end = start + timedelta(days=30)

    # extracts a fraction of records between a start and end date
    records = pd.DataFrame()
    pd.options.mode.chained_assignment = None
    num_batches = math.ceil((end - start).days / 30)
    for i in range(1, num_batches+1):
        start_week = math.ceil(((batch_start - date(year=batch_start.year, month=1, day=1)).days / 365) * 52)
        start_year_week = int(batch_start.year.__str__() + start_week.__str__().zfill(2))

        end_week = math.ceil(((batch_end - date(year=batch_end.year, month=1, day=1)).days / 365) * 52)
        end_year_week = int(batch_end.year.__str__() + end_week.__str__().zfill(2))

        print(f"Running batch {i}/{num_batches}: From year/week {start_year_week} to year/week {end_year_week}")
        results = query(db, start_year_week, min(now_year_week, end_year_week), frac)
        print(f"{len(results)} records found...")
        # Only preprocess if batch has records
        if len(results) > 0:
            print(f"Preprocessing batch...")
            records = records.append(preprocess(results), ignore_index=True)

        batch_start = batch_end
        batch_end += timedelta(days=30)
        print("Batch completed. {:.2f}% batches completed. \n".format((i/num_batches)*100))

    print(f"{len(records)} records extracted and preprocessed")
    utilities.print_elapsed_time(extraction_start_time)
    return records


def create_filename(frac):
    file_id = utilities.get_timestamp()
    fraction = utilities.string_to_decimal(frac)
    filename = f"extract_frac-{int(fraction*100)}_{file_id}"
    return filename


def store(records, frac):
    """
    Stores results from each query into a pickle
    """
    print("Saving records...")
    start_time = time.time()
    filename = create_filename(frac)
    output_path = os.path.join(paths.data_extracted_dir, filename)
    records.to_csv(output_path + ".csv")

    utilities.print_elapsed_time(start_time)
    return output_path + ".csv"


if __name__ == '__main__':
    start, end, frac = ['2018-06-01', '2019-05-31', .01]

    print("Querying and preprocessing records ...")
    records = batch_query(start, end, frac)
    print("{} records extracted and preprocessed".format(len(records)))

    print("Storing records ...")
    output_path = store(records, frac)
    print(f"{len(records)} records stored as {output_path}")
