import time
from datetime import datetime
from datetime import timedelta
from decimal import Decimal


def get_timestamp():
    """
    Create a id using the current timestamp

    Returns:
            str: Current timestamp in YYMMDD-HHMM format
    """
    now = datetime.now()
    model_id = now.strftime("%Y%m%d-%H%M")
    return model_id


def print_elapsed_time(start_time):
    """
    Prints amount of time elapsed since start time

    Args:
            start_time (datetime): timestamp
    """
    elapsed_time_secs = time.time() - start_time
    msg = "Execution took: %s secs (Wall clock time) \n" % timedelta(seconds=round(elapsed_time_secs))
    print(msg)


def string_to_decimal(string):
    # Convert fraction string to float
    try:
        decimal = Decimal(string)
    except ValueError:
        print("No decimal found, using default=1")
        decimal = 1
    return decimal
