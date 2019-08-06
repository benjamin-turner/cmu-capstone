"""
Benchmarking Script

This is the key script that generates:
    1. Terminal output 
    2. Excel files for offline review
    
This file should be imported as a module and contains the following function used in main.py:
    
    * get_selected_metrics - get all selected metrics requested by user
    
"""
import os
import time
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import menu_options
import paths

'''
Store the current directory in a variable
'''
orig_dir = os.getcwd()
# Retrieve raw scores from the full SS matrix
def get_raw_scores(sid, preloaded_matrix):
    '''
    Get raw similarity scores from the preloaded similarity matrix 
    
    Args:
        sid (str): sid for which raw scores are needed
        preloaded_matrix (pandas dataframe object): Raw similarity score matrix
    
    Returns:
        clean_scores (numpy array): Cleaned scores after removing nans and negative values
        
    '''
    full_ss_matrix_optimized = np.array(preloaded_matrix)
    full_matrix_indices = np.array(preloaded_matrix.index.values)

    # Retrieve the index at which the requested sid is present in the SS matrix
    test_sid_idx = np.where(full_matrix_indices == sid)[0][0]

    # Return the SS array
    raw_score_array = full_ss_matrix_optimized[test_sid_idx]

    # Change NANs to 0
    raw_score_array[np.isnan(raw_score_array)] = 0

    # Identify indices where scores <=0 are currently present
    dirty_score_idx = np.where(raw_score_array <= 0)

    # Update these indices with a different value (0.0001 is chosen here)
    clean_scores = raw_score_array
    np.put(clean_scores, dirty_score_idx, np.full(shape=len(dirty_score_idx), fill_value=0.0001))

    return np.array(clean_scores)


def ss_threshold(sid, percentile, preloaded_matrix):
    '''
    Get threshold score based on user-selected percentile

    For a given sid and percentile, this function returns the threshold similarity score value.
    All sids with a similarity score higher than the threshold value will be deemed 'similar customers'
    at the given percentile(default 90th %ile) level
    
    Args:
        sid (str): sid for which threshold score is needed
        percentile (int): percentile value between 0-100 provided by the user
        preloaded_matrix (pandas dataframe object): Raw similarity score matrix
    
    Returns:
        threshold_score (float): Threshold score for selected sid at given percentile
        
    '''
    threshold_score = (np.percentile(get_raw_scores(sid, preloaded_matrix), percentile))
    return np.round(threshold_score, 3)


def get_similar_customers(input_sid, percentile, user, preloaded_matrix):
    '''
    Get similar customers and their similarity scores based on input sid and percentile.

    This directly returns a list of ALL similar customers that are above a certain percentile threshold(default 90th %ile).

    Args:
        input_sid (str): sid for which similar customers are needed
        percentile (int): Value between 0-100 provided by user
        user (int): if 1, print top 10 most similar sids ranked by similarity scoree
        preloaded_matrix (pandas dataframe object): Raw similarity score matrix
    
    Returns:
        sorted_sids (numpy array): Array containing sids sorted in descending order
        df (pandas dataframe object): Dataframe containing both sids and corresponding similarity score
        
    '''
    full_ss_matrix_optimized = np.array(preloaded_matrix)
    full_matrix_indices = np.array(preloaded_matrix.index.values)
    
    inp_sid = input_sid
    # Get threshold score for requested sid
    threshold_score = ss_threshold(inp_sid, percentile, preloaded_matrix)
    # Get index of requested sid in the full_ss_matrix
    sid_idx = np.where(full_matrix_indices == inp_sid)[0][0]
    # Get raw scores for the test_sid from the full_ss_matrix
    sid_scores = get_raw_scores(full_matrix_indices[sid_idx], preloaded_matrix)
    # First get all scores that are higher than the threshold score
    similar_scores = full_ss_matrix_optimized[sid_idx][np.where(sid_scores >= threshold_score)[0]]
    # Then get a list of sids that have scores higher than the threshold score
    similar_sids = full_matrix_indices[np.where(sid_scores >= threshold_score)[0]]
    # Sort scores in descending order
    sorted_idx = np.argsort(similar_scores)[::-1]
    # Sort similar_sids in descending order based on scores
    sorted_sids = similar_sids[sorted_idx]
    # Sort similar_scores in descending order based on scores
    sorted_scores = similar_scores[sorted_idx]
    if (user == 1):
        print('The top 10 similar SIDs are:')
        print(sorted_sids[0:10], '\n')
        print(inp_sid, 'has', len(sorted_sids), f'similar customers in the top {100 - percentile}%\n')
    cols = ["Similar_SIDs", "Similarity_Score"]
    KPIdata = np.column_stack((sorted_sids, sorted_scores))
    df = pd.DataFrame(data=KPIdata, columns=cols)
    return sorted_sids, df

#############################################################################################################
#Begin KPI Calculation

#code below is for testing
#input_sid = '7B8E9E45F5'
#percentile = 90
#user = 1
#df.head()

############################################################################################################

# Average Spend KPI (Option 1)
def KPI_avg_spend(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    '''
    Get all KPIs related to average spend
    Average total spend:
            a. Average total spend
            b. Average total spend per shipper
            c. Average total spend per lb 
            d. Average total spend per lb per shipper
    
    Args:
        input_sid (str): sid for which similar customers are needed
        percentile (int): Value between 0-100 provided by user
        user (int): if 1, print top 10 most similar sids ranked by similarity scoree
        preloaded_matrix (pandas dataframe object): Raw similarity score matrix
        preloaded_KPIs (pandas dataframe object): KPI database
    
    Returns:
        df (pandas dataframe object): Dataframe containing avg_spend KPIs
        
    '''
    
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    avg_spend_df = np.round(preloaded_KPIs.iloc[:, 0:6], 2)
     
    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix)[0]
    similar_customers_idx = np.isin(full_KPI_indices, similar_customer_list)

    self_avg_spend = np.array(np.round(avg_spend_df.loc[input_sid], 2))
    peer_avg_spend = np.round(np.mean(np.array(avg_spend_df)[similar_customers_idx], axis=0), 2)

    cols = ['Metric', 'Self', 'Peers']
    metrics = avg_spend_df.columns.tolist()
    self_vals = self_avg_spend
    peer_vals = peer_avg_spend
    if (user == 1):
        t = PrettyTable(cols)
        for i in range(len(metrics)):
            row = [metrics[i], self_vals[i], peer_vals[i]]
            t.add_row(row)
        print("--- Average Spend Metrics ---")
        print(" ")
        print(t)
        print(" ")
        print("----------------------------------------------------------")
        print(" ")
    # create result dataframe
    cols = ["Self", "Peers"]
    KPIdata = np.column_stack((self_vals, peer_vals))
    df = pd.DataFrame(data=KPIdata, columns=cols, index=metrics)
    return df

# Average Spend per Shipping Method KPI (Option 2)
def KPI_avg_spend_per_method(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    '''
    Get all KPIs related to average spend per shipping method
    Average spend per shipping method:
            a. Average spend per method
            b. Average spend per method per lb
            c. Average spend per method per shipper
            d. Average spend per method per lb per shipper
    
    Args:
        input_sid (str): sid for which similar customers are needed
        percentile (int): Value between 0-100 provided by user
        user (int): if 1, print top 10 most similar sids ranked by similarity scoree
        preloaded_matrix (pandas dataframe object): Raw similarity score matrix
        preloaded_KPIs (pandas dataframe object): KPI database
    
    Returns:
        df (pandas dataframe object): Dataframe containing relevant KPIs
        
    '''
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    spend_per_method_df = np.round(preloaded_KPIs.iloc[:, 6:44], 2)

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix)[0]
    similar_customers_idx = np.isin(full_KPI_indices, similar_customer_list)

    self_spend_per_method = np.array(np.round(spend_per_method_df.loc[input_sid], 2))
    peer_spend_per_method = np.round(np.mean(np.array(spend_per_method_df)[similar_customers_idx], axis=0), 2)

    # Print output
    cols = ['Metric', 'Self', 'Peers']
    metrics = spend_per_method_df.columns.tolist()
    self_vals = self_spend_per_method
    peer_vals = peer_spend_per_method
    if (user == 1):
        t = PrettyTable(cols)
        for i in range(len(metrics)):
            row = [metrics[i], self_vals[i], peer_vals[i]]
            t.add_row(row)
        print("--- Average Spend per Shipping Method ---")
        print(" ")
        print(t)
        print(" ")
        print("----------------------------------------------------------")
        print(" ")
    # create result dataframe
    cols = ["Self", "Peers"]
    KPIdata = np.column_stack((self_vals, peer_vals))
    df = pd.DataFrame(data=KPIdata, columns=cols, index=metrics)
    return df


# Average Spend per Month KPI (Option 3)
def KPI_spend_per_month(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    '''
    Get all KPIs related to average spend per month
    Average spend per month:
            a. Average spend per month
            b. Average spend per month per lb
            c. Average spend per month per shipper
            d. Average spend per month per lb per shipper
    Args:
        input_sid (str): sid for which similar customers are needed
        percentile (int): Value between 0-100 provided by user
        user (int): if 1, print top 10 most similar sids ranked by similarity scoree
        preloaded_matrix (pandas dataframe object): Raw similarity score matrix
        preloaded_KPIs (pandas dataframe object): KPI database
    
    Returns:
        df (pandas dataframe object): Dataframe containing relevant KPIs
        
    '''
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    spend_per_month_df = np.round(preloaded_KPIs.iloc[:, 44:116], 2)
    
    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix)[0]
    similar_customers_idx = np.isin(full_KPI_indices, similar_customer_list)
    
    self_spend_per_month = np.array(np.round(spend_per_month_df.loc[input_sid], 2))
    peer_spend_per_month = np.round(np.mean(np.array(spend_per_month_df)[similar_customers_idx], axis=0), 2)
    # Print output
    cols = ['Metric', 'Self', 'Peers']
    metrics = spend_per_month_df.columns.tolist()
    self_vals = self_spend_per_month
    peer_vals = peer_spend_per_month
    if (user == 1):
        t = PrettyTable(cols)
        for i in range(len(metrics)):
            row = [metrics[i], self_vals[i], peer_vals[i]]
            t.add_row(row)
        print("--- Average Spend per Month ---")
        print(" ")
        print(t)
        print(" ")
        print("----------------------------------------------------------")
        print(" ")
    # create result dataframe
    cols = ["Self", "Peers"]
    KPIdata = np.column_stack((self_vals, peer_vals))
    df = pd.DataFrame(data=KPIdata, columns=cols, index=metrics)
    return df


# Average Discount KPI (Option 4)
def KPI_avg_discount(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    '''
    Get all KPIs related to average discount
    Average total discounts:
            a. Average total discount
            b. Average total discount per shipper
            c. Average total discount per lb 
            d. Average total discount per lb per shipper
    
    Args:
        input_sid (str): sid for which similar customers are needed
        percentile (int): Value between 0-100 provided by user
        user (int): if 1, print top 10 most similar sids ranked by similarity scoree
        preloaded_matrix (pandas dataframe object): Raw similarity score matrix
        preloaded_KPIs (pandas dataframe object): KPI database
    
    Returns:
        df (pandas dataframe object): Dataframe containing relevant KPIs
        
    '''
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    avg_discount_df = np.round(preloaded_KPIs.iloc[:,116:122], 2)

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix)[0]
    similar_customers_idx = np.isin(full_KPI_indices, similar_customer_list)

    self_avg_discount = np.array(np.round(avg_discount_df.loc[input_sid], 2))
    peer_avg_discount = np.round(np.mean(np.array(avg_discount_df)[similar_customers_idx], axis=0), 2)
    # Print output
    cols = ['Metric', 'Self', 'Peers']
    metrics = avg_discount_df.columns.tolist()
    self_vals = self_avg_discount
    peer_vals = peer_avg_discount
    if (user == 1):
        t = PrettyTable(cols)
        for i in range(len(metrics)):
            row = [metrics[i], self_vals[i], peer_vals[i]]
            t.add_row(row)
        print("--- Average Discount Metrics ---")
        print(" ")
        print(t)
        print(" ")
        print("----------------------------------------------------------")
        print(" ")
    # create result dataframe
    cols = ["Self", "Peers"]
    KPIdata = np.column_stack((self_avg_discount, peer_avg_discount))
    df = pd.DataFrame(data=KPIdata, columns=cols, index=metrics)
    return df

# Average Discount per Shipping Method KPI (Option 5)
def KPI_avg_discount_per_method(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    '''
    Get all KPIs related to average discount per shipping method
    Average discount per shipping method:
            a. Average discount per method
            b. Average discount per method per lb
            c. Average discount per method per shipper
            d. Average discount per method per lb per shipper
    
    Args:
        input_sid (str): sid for which similar customers are needed
        percentile (int): Value between 0-100 provided by user
        user (int): if 1, print top 10 most similar sids ranked by similarity scoree
        preloaded_matrix (pandas dataframe object): Raw similarity score matrix
        preloaded_KPIs (pandas dataframe object): KPI database
    
    Returns:
        df (pandas dataframe object): Dataframe containing relevant KPIs
        
    '''
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    discount_per_method_df = np.round(preloaded_KPIs.iloc[:, 122:160], 2)

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix)[0]
    similar_customers_idx = np.isin(full_KPI_indices, similar_customer_list)

    self_discount_per_method = np.array(np.round(discount_per_method_df.loc[input_sid], 2))
    peer_discount_per_method = np.round(np.mean(np.array(discount_per_method_df)[similar_customers_idx], axis=0), 2)

    # Print output
    cols = ['Metric', 'Self', 'Peers']
    metrics = discount_per_method_df.columns.tolist()
    self_vals = self_discount_per_method
    peer_vals = peer_discount_per_method
    if (user == 1):
        t = PrettyTable(cols)
        for i in range(len(metrics)):
            row = [metrics[i], self_vals[i], peer_vals[i]]
            t.add_row(row)
        print("--- Average Discount per Shipping Method ---")
        print(" ")
        print(t)
        print(" ")
        print("----------------------------------------------------------")
        print(" ")
    # create result dataframe
    cols = ["Self", "Peers"]
    KPIdata = np.column_stack((self_vals, peer_vals))
    df = pd.DataFrame(data=KPIdata, columns=cols, index=metrics)
    return df

# Discounts per Zone KPI (Option 6)
def KPI_discount_per_zone(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    '''
    Get all KPIs related to average discount per zone
    Average discounts per zone
            a. Average discount per zone
            b. Average discount per zone per lb
            c. Average discount per zone per shipper
            d. Average discount per zone per lb per shipper 
    
    Args:
        input_sid (str): sid for which similar customers are needed
        percentile (int): Value between 0-100 provided by user
        user (int): if 1, print top 10 most similar sids ranked by similarity scoree
        preloaded_matrix (pandas dataframe object): Raw similarity score matrix
        preloaded_KPIs (pandas dataframe object): KPI database
    
    Returns:
        df (pandas dataframe object): Dataframe containing relevant KPIs
        
    '''
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    discount_per_zone_df = np.round(preloaded_KPIs.iloc[:, 192:], 2)

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix)[0]
    similar_customers_idx = np.isin(full_KPI_indices, similar_customer_list)

    self_discount_per_zone = np.array(np.round(discount_per_zone_df.loc[input_sid], 2))
    peer_discount_per_zone = np.round(np.mean(np.array(discount_per_zone_df)[similar_customers_idx], axis=0), 2)

    # Print output
    cols = ['Metric', 'Self', 'Peers']
    metrics = discount_per_zone_df.columns.tolist()
    self_vals = self_discount_per_zone
    peer_vals = peer_discount_per_zone
    if (user == 1):
        t = PrettyTable(cols)
        for i in range(len(metrics)):
            row = [metrics[i], self_vals[i], peer_vals[i]]
            t.add_row(row)
        print("--- Average Discount per Zone ---")
        print(" ")
        print(t)
        print(" ")
        print("----------------------------------------------------------")
        print(" ")
    # create result dataframe
    cols = ["Self", "Peers"]
    # idx =   volume_per_method_df.columns.tolist()
    KPIdata = np.column_stack((self_vals, peer_vals))
    df = pd.DataFrame(data=KPIdata, columns=cols, index=metrics)
    return df


# Shipping Method Proportion KPI (Option 7)
def KPI_methodwise_proportion(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    '''
    Get all KPIs related to methodwise proportion
    Proportion of shipping methods used
            a. Average volume proportion per method
    
    Args:
        input_sid (str): sid for which similar customers are needed
        percentile (int): Value between 0-100 provided by user
        user (int): if 1, print top 10 most similar sids ranked by similarity scoree
        preloaded_matrix (pandas dataframe object): Raw similarity score matrix
        preloaded_KPIs (pandas dataframe object): KPI database
    
    Returns:
        df (pandas dataframe object): Dataframe containing relevant KPIs
        
    '''
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    methodwise_proportion_df = np.round(preloaded_KPIs.iloc[:, 160:167], 2)

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix)[0]
    similar_customers_idx = np.isin(full_KPI_indices, similar_customer_list)

    self_prop_per_method = np.array(np.round(methodwise_proportion_df.loc[input_sid], 2)) * 100
    peer_prop_per_method = np.round(np.mean(np.array(methodwise_proportion_df)[similar_customers_idx], axis=0), 2) * 100

    # Print output
    cols = ['Metric', 'Self', 'Peers']
    metrics = methodwise_proportion_df.columns.tolist()
    self_vals = self_prop_per_method
    peer_vals = peer_prop_per_method
    if (user == 1):
        t = PrettyTable(cols)
        for i in range(len(metrics)):
            row = [metrics[i], self_vals[i], peer_vals[i]]
            t.add_row(row)
        print("--- Shipping Method Proportion ---")
        print(" ")
        print(t)
        print(" ")
        print("----------------------------------------------------------")
        print(" ")
    # create result dataframe
    cols = ["Self", "Peers"]
    # idx = metrics
    KPIdata = np.column_stack((self_vals, peer_vals))
    df = pd.DataFrame(data=KPIdata, columns=cols, index=metrics)

    return df

# Shipper Proportion KPI (Option 8)
def KPI_shipper_proportion(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    '''
    Get all KPIs related to average shipper proportion
    Proportion of each carrier used
            a. Proportional carrier use(%FedEx, %UPS)
    
    Args:
        input_sid (str): sid for which similar customers are needed
        percentile (int): Value between 0-100 provided by user
        user (int): if 1, print top 10 most similar sids ranked by similarity scoree
        preloaded_matrix (pandas dataframe object): Raw similarity score matrix
        preloaded_KPIs (pandas dataframe object): KPI database
    
    Returns:
        df (pandas dataframe object): Dataframe containing relevant KPIs
        
    '''
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    fedex_prop_list = np.array(np.round(preloaded_KPIs['proportion_fedex'], 2))
    ups_prop_list = np.array(np.round(preloaded_KPIs['proportion_ups'], 2))

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix)[0]
    similar_customers_idx = np.isin(full_KPI_indices, similar_customer_list)

    self_fedex_proportion = np.round(preloaded_KPIs.loc[input_sid][186], 2) * 100
    self_ups_proportion = np.round(preloaded_KPIs.loc[input_sid][187], 2) * 100

    peer_fedex_proportion = np.round(np.mean(fedex_prop_list[similar_customers_idx]), 2) * 100
    peer_ups_proportion = np.round(np.mean(ups_prop_list[similar_customers_idx]), 2) * 100

    if (user == 1):
        # Print output
        t = PrettyTable(['SID', '%Fedex', '%UPS'])
        row1 = [input_sid, self_fedex_proportion, self_ups_proportion]
        row2 = ['Peers', peer_fedex_proportion, peer_ups_proportion]
        t.add_row(row1)
        t.add_row(row2)
        print("--- Shipper Proportion ---")
        print(" ")
        print(t)
        print(" ")
        print("----------------------------------------------------------")
        print(" ")

    # create result dataframe
    cols = ["Self", "Peers"]
    idx = ["%FedEx", "%UPS"]
    KPIdata = np.array([[self_fedex_proportion, peer_fedex_proportion],
                        [self_ups_proportion, peer_ups_proportion]])
    df = pd.DataFrame(data=KPIdata, columns=cols, index=idx)
    return df

# Volume per method KPI (Option 9)
def KPI_volume_per_method(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    '''
    Get all KPIs related to average volume per method
    Total volume shipper per shipping method
            a. Average volume per method
            b. Average volume per method per shipper
    
    Args:
        input_sid (str): sid for which similar customers are needed
        percentile (int): Value between 0-100 provided by user
        user (int): if 1, print top 10 most similar sids ranked by similarity scoree
        preloaded_matrix (pandas dataframe object): Raw similarity score matrix
        preloaded_KPIs (pandas dataframe object): KPI database
    
    Returns:
        df (pandas dataframe object): Dataframe containing relevant KPIs
        
    '''
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    volume_per_method_df = np.round(preloaded_KPIs.iloc[:, 167:186], 2)

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix)[0]
    similar_customers_idx = np.isin(full_KPI_indices, similar_customer_list)

    self_volume_per_method = np.array(np.round(volume_per_method_df.loc[input_sid], 2))
    peer_volume_per_method = np.round(np.mean(np.array(volume_per_method_df)[similar_customers_idx], axis=0), 2)

    # Print output
    cols = ['Metric', 'Self', 'Peers']
    metrics = volume_per_method_df.columns.tolist()
    self_vals = self_volume_per_method
    peer_vals = peer_volume_per_method
    if (user == 1):
        t = PrettyTable(cols)
        for i in range(len(metrics)):
            row = [metrics[i], self_vals[i], peer_vals[i]]
            t.add_row(row)
        print("--- Volume Shipped per Shipping Method ---")
        print(" ")
        print(t)
        print(" ")
        print("----------------------------------------------------------")
        print(" ")
    # create result dataframe
    cols = ["Self", "Peers"]
    # idx =   volume_per_method_df.columns.tolist()
    KPIdata = np.column_stack((self_vals, peer_vals))
    df = pd.DataFrame(data=KPIdata, columns=cols, index=metrics)

    return df

metrics = menu_options.benchmarking_metric_options
metric_to_function = {
    metrics[0]: KPI_avg_spend,
    metrics[1]: KPI_avg_spend_per_method,
    metrics[2]: KPI_spend_per_month,
    metrics[3]: KPI_avg_discount,
    metrics[4]: KPI_avg_discount_per_method,
    metrics[5]: KPI_discount_per_zone,
    metrics[6]: KPI_methodwise_proportion,
    metrics[7]: KPI_shipper_proportion,
    metrics[8]: KPI_volume_per_method,
}

def get_descriptors_for_similar_sids(input_sid, percentile, user, preloaded_matrix,preloaded_KPIs):
    '''
    Get descriptors for similar sids, such as industry and most common sender/recipient state
    
    Args:
        input_sid (str): sid for which similar customers are needed
        percentile (int): Value between 0-100 provided by user
        user (int): if 1, print top 10 most similar sids ranked by similarity scoree
        preloaded_matrix (pandas dataframe object): Raw similarity score matrix
        preloaded_KPIs (pandas dataframe object): KPI database
    
    Returns:
        df (pandas dataframe object): Dataframe containing relevant descriptors
        
    '''
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    descriptors_df = np.round(preloaded_KPIs.iloc[:, 188:192], 2)

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix=preloaded_matrix)[0]
    similar_customers_idx = np.isin(full_KPI_indices, similar_customer_list)

    peer_descriptors = descriptors_df.iloc[similar_customers_idx]
    
    return peer_descriptors

def get_selected_metrics(selected_metrics, sid, percentile, preloaded_matrix, preloaded_KPIs):
    '''
    Get all selected metrics requested by user
    This function also stores all KPIs(even if not requested) and similar sids to an excel file per execution
    
    Args:
        selected_metrics (list): list of metrics selected by user
        sid (str): sid for which similar customers are needed
        percentile (int): Value between 0-100 provided by user
        user (int): if 1, print top 10 most similar sids ranked by similarity scoree
        preloaded_matrix (pandas dataframe object): Raw similarity score matrix
        preloaded_KPIs (pandas dataframe object): KPI database
    
    Returns:
        True (bool): Dummy response. All terminal output and file generation is handled within the function
        
    '''
    dispatch_list = []
    for metric in menu_options.benchmarking_metric_options:
        # If metric is selected: add function that corresponds to metric to dispatch list
        if metric in selected_metrics:
            dispatch_list.append(metric_to_function[metric](sid, percentile, user=1,
                                                            preloaded_matrix=preloaded_matrix,
                                                            preloaded_KPIs=preloaded_KPIs))

    #Print all KPIs to Excel
    excel_list = []
    
    for metric in menu_options.benchmarking_metric_options:
        excel_list.append(metric_to_function[metric](sid, percentile, user=0,
                                                            preloaded_matrix=preloaded_matrix,
                                                            preloaded_KPIs=preloaded_KPIs))
    allKPIs = pd.concat(excel_list)


    # Get a list of similar sids and respective similarity scores
    similarity_data = get_similar_customers(sid, percentile, preloaded_matrix=preloaded_matrix,user = 0)[1]
    similarity_data = similarity_data.set_index('Similar_SIDs')
    
    user = 0
    descriptors = get_descriptors_for_similar_sids(sid, percentile,user, preloaded_matrix=preloaded_matrix,preloaded_KPIs=preloaded_KPIs)

    merged_df = pd.merge(similarity_data, descriptors, how = 'inner', left_index = True, right_index = True)

    # filename convention: benchmark_<SID>_<PERCENTILE>_<YYYYMMDD-HHMM>
    output_path = os.path.join(orig_dir, paths.output_benchmarking_dir)
    timestr = time.strftime("%Y%m%d-%H%M")

    file_name = output_path + '\Benchmarking_' + str(sid) + '_' + str(
        percentile) + '_' + timestr + '.xlsx'

    os.getcwd()
    with pd.ExcelWriter(file_name) as writer:
        allKPIs.to_excel(writer, sheet_name='KPIs')
        merged_df.to_excel(writer, sheet_name='Reference_Data')
    return True