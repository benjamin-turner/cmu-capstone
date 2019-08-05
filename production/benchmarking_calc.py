"""
Benchmarking Preprocessing Calculations Script

This script contains the functions to preprocess data in the format needed for
creating a Similarity Score matrix and KPI Database

Essentially, this script generates both the similarity and dissimilarity metrics for a given set of sids

This file should be imported as a module and contains the following functions that are used in main.py:
    
    * create_similarity_score_matrix - preprocess data, generate similarity scores and corresponding matrix
    * create_customer_KPI_database - preprocess data and calculate relevant KPIs 

"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import paths
import utilities
from tqdm import tqdm
warnings.filterwarnings('ignore')

'''
Store the current directory in a variable
'''
orig_dir = os.getcwd()

#Begin Benchmarking-specific preprocessing
#Assumption - data in extract is only for last 12 months

def get_relevant_features(df):
    '''
    This function extracts the relevant features needed to start building out the benchmarking solution
    
    Args:
        df (pandas dataframe object): Raw dataframe. This will ideally be an extract from the 71lbs database
    
    Returns:
        benchmarking_df (pandas dataframe object): Initialized dataframe with relevant attributes
        
    '''
    benchmarking_df = df[['business_sid','weight', 'zone', 'shipment_date']]
    benchmarking_df['ship_month'] = pd.DatetimeIndex(benchmarking_df['shipment_date']).month
    reqrd_cols = ['business_sid', 'weight', 'zone', 'ship_month']
    return benchmarking_df[reqrd_cols]

#Create aggregated/pivot tables for the following 6 metrics:
#1. Total Volume
#2. Volume proportion per Zone
#3. Volume proportion per Month
#4. Total Weight
#5. Weight proportion per Zone
#6. Weight proportion per Month
#Create pivot table for total volume
####################################################################################################################        
def create_totalVolume_Table(arg_df):
    '''
    This function creates a dataframe listing the total volume(# of shipments) by each sid present
    in the input dataframe
    
    Args:
        df (pandas dataframe object): Input dataframe. This will contain all relevant benchmarking attributes
    
    Returns:
        total_vol_Table (pandas dataframe object): Dataframe containing total shipment volume for each sid
        
    '''
    total_vol_Table = pd.DataFrame(arg_df.groupby(['business_sid']).count().iloc[:,0])
    total_vol_Table.columns = ['Total_Volume']
    return total_vol_Table
####################################################################################################################    
def create_totalWeight_Table(arg_df):
    '''
    This function creates a dataframe listing the total weight of shipments for each sid present
    in the input dataframe
    
    Args:
        df (pandas dataframe object): Input dataframe. This will contain all relevant benchmarking attributes
    
    Returns:
        total_weight_Table (pandas dataframe object): Dataframe containing total shipment weight for each sid
        
    '''
    total_weight_Table =  pd.DataFrame(arg_df.groupby(['business_sid']).sum().iloc[:,0])
    total_weight_Table.columns = ['Total_Weight']
    return total_weight_Table
####################################################################################################################    
def create_volumePerMonth_pivotTable(arg_df):
    '''
    This function creates a pivot table listing the total volume(# of shipments) for each month,
    for each sid present in the input dataframe
    
    Args:
        df (pandas dataframe object): Input dataframe. This will contain all relevant benchmarking attributes
    
    Returns:
        volmonthdf_pivotTable (pandas dataframe object): Pivot table containing total shipment volume per month for each sid
    '''
    volmonthdf = pd.DataFrame(arg_df.groupby(['business_sid', 'ship_month']).count().iloc[:,0])
    volmonthdf.columns = ['Count']
    volmonthdf_pivotTable = pd.pivot_table(volmonthdf, values = 'Count', index = ['business_sid'], columns = 'ship_month', 
                     aggfunc = np.sum, fill_value = 0)
    return volmonthdf_pivotTable
####################################################################################################################    
def create_weightPerMonth_pivotTable(arg_df):
    '''
    This function creates a pivot table listing the total weight of shipments for each month,
    for each sid present in the input dataframe
    
    Args:
        df (pandas dataframe object): Input dataframe. This will contain all relevant benchmarking attributes
    
    Returns:
        weightmonthdf_pivotTable (pandas dataframe object): Pivot table containing total shipment weight per month for each sid
    '''
    weightmonthdf = pd.DataFrame(arg_df.groupby(['business_sid', 'ship_month']).sum().iloc[:,0])
    weightmonthdf_pivotTable = pd.pivot_table(weightmonthdf, values = 'weight', index = ['business_sid'], columns = 'ship_month', 
                     aggfunc = np.sum, fill_value = 0)
    return weightmonthdf_pivotTable
####################################################################################################################    
#Create pivot table for volume proportions
def create_volumeProportion_pivotTable(arg_df):
    '''
    This function creates a pivot table listing the proportional volume of total shipments per zone,
    for each sid present in the input dataframe
    
    Args:
        df (pandas dataframe object): Input dataframe. This will contain all relevant benchmarking attributes
    
    Returns:
        volumeProportion_pivotTable (pandas dataframe object): Pivot table containing shipment proportion per zone for each sid
    '''
    tempvoldf = pd.DataFrame(arg_df.groupby(['business_sid', 'zone']).count().iloc[:,0])
    tempvoldf.columns = ['Count']
    volumeCount_pivotTable = pd.pivot_table(tempvoldf, values = 'Count', index = ['business_sid'], columns = 'zone', 
                     aggfunc = np.sum, fill_value = 0)    
    volumeProportion_pivotTable = volumeCount_pivotTable.apply(lambda x: x/x.sum(), axis = 1)
    return volumeProportion_pivotTable
####################################################################################################################    
#Create pivot table for weight proportions
def create_weightProportion_pivotTable(arg_df):
    '''
    This function creates a pivot table listing the proportional weight of total shipments per zone,
    for each sid present in the input dataframe
    
    Args:
        df (pandas dataframe object): Input dataframe. This will contain all relevant benchmarking attributes
    
    Returns:
        weightProportion_pivotTable (pandas dataframe object): Pivot table containing shipment weight proportion per zone for each sid
    '''
    tempwtdf = arg_df.groupby(['business_sid', 'zone']).sum()
    weightSum_pivotTable = pd.pivot_table(tempwtdf, values = 'weight', index = ['business_sid'], columns = 'zone',
                                         aggfunc = np.sum, fill_value = 0)
    weightProportion_pivotTable= weightSum_pivotTable.apply(lambda x: x/x.sum(), axis = 1)
    return weightProportion_pivotTable
####################################################################################################################    
#Combine all pivot tables under a single function
#This will come in handy later while calculating similarity scores
def create_PivotTables(arg_df):
    '''
    Create all pivotTables in a single function
    
    Args:
        df (pandas dataframe object): Input dataframe. This will contain all relevant benchmarking attributes
    
    Returns:
        6 dataframes(pandas dataframe object): Each of the 6 pivot table necessary for creating the similarity score
    '''
    a = create_totalVolume_Table(arg_df)
    b = create_totalWeight_Table(arg_df)
    c = create_volumePerMonth_pivotTable(arg_df)
    d = create_weightPerMonth_pivotTable(arg_df)
    e = create_volumeProportion_pivotTable(arg_df)
    f = create_weightProportion_pivotTable(arg_df)
    return a,b,c,d,e,f

####################################################################################################################    
### Calculate the indices required as input for the Similarity Score calculation
def calculate_volumetric_scale_index(sid_in_focus, volumeTable, sid_count, sid_list):
    '''
    Creates the volumetric scale index(VSI). Index 1 of 6 for calculating Similarity Scores
    
    Args:
        sid_in_focus (str): sid for which the index is calculated
        volumeTable (pandas dataframe object): dataframe with relevant values for computing VSI
        sid_count (int): # of total sids against which VSI is computed
        sid_list (list): list of all sids against which VSI is computed
    
    Returns:
        vsi_list (numpy array): array containing VSI values for the input sid_in_focus
    '''
    vsi_list = np.empty(sid_count)
    sid_in_focus_arr = np.array(volumeTable.loc[sid_in_focus])
    comparison_arr = np.array(volumeTable)
    for x in range(sid_count):
        if(sid_list[x]!=sid_in_focus):
            vsi_numerator = np.minimum(sid_in_focus_arr,comparison_arr[x]).sum()
            vsi_denominator = np.maximum(sid_in_focus_arr,comparison_arr[x]).sum()
            vsi_list[x] = vsi_numerator/vsi_denominator
        else:
            vsi_list[x] = -1  # return a value of -1 when comparing a customer to themselves        
    return vsi_list

def calculate_weight_scale_index(sid_in_focus, weightTable, sid_count, sid_list):
    '''
    Creates the weight scale index(WSI). Index 2 of 6 for calculating Similarity Scores
    
    Args:
        sid_in_focus (str): sid for which the index is calculated
        volumeTable (pandas dataframe object): dataframe with relevant values for computing WSI
        sid_count (int): # of total sids against which WSI is computed
        sid_list (list): list of all sids against which WSI is computed
    
    Returns:
        wsi_list (numpy array): array containing WSI values for the input sid_in_focus
    '''
    wsi_list = np.empty(sid_count)
    sid_in_focus_arr = np.array(weightTable.loc[sid_in_focus])
    comparison_arr = np.array(weightTable)
    for x in range(sid_count):
        if(sid_list[x]!=sid_in_focus):
            wsi_numerator = np.minimum(sid_in_focus_arr,comparison_arr[x]).sum()
            wsi_denominator = np.maximum(sid_in_focus_arr,comparison_arr[x]).sum()
            wsi_list[x] = wsi_numerator/wsi_denominator
        else:
            wsi_list[x] = -1  # return a value of -1 when comparing a customer to themselves        
    return wsi_list

def calculate_volume_per_month_index(sid_in_focus, volumePerMonthTable, sid_count, sid_list):
    '''
    Creates the volume per month index(VPMI). Index 3 of 6 for calculating Similarity Scores
    
    Args:
        sid_in_focus (str): sid for which the index is calculated
        volumeTable (pandas dataframe object): dataframe with relevant values for computing VPMI
        sid_count (int): # of total sids against which VPMI is computed
        sid_list (list): list of all sids against which VPMI is computed
    
    Returns:
        vpmi_list (numpy array): array containing VSI values for the input sid_in_focus
    '''
    vpmi_list = np.empty(sid_count)
    sid_in_focus_arr = np.array(volumePerMonthTable.loc[sid_in_focus])
    comparison_arr = np.array(volumePerMonthTable)
    for x in range(sid_count):
        if(sid_list[x]!=sid_in_focus):
            vpmi_numerator = np.minimum(sid_in_focus_arr,comparison_arr[x]).sum()
            vpmi_denominator = np.maximum(sid_in_focus_arr,comparison_arr[x]).sum()
            vpmi_list[x] = vpmi_numerator/vpmi_denominator
        else:
            vpmi_list[x] = -1  # return a value of -1 when comparing a customer to themselves        
    return vpmi_list

def calculate_weight_per_month_index(sid_in_focus, weightPerMonthTable, sid_count, sid_list):
    '''
    Creates the weight per month index(WPMI). Index 4 of 6 for calculating Similarity Scores
    
    Args:
        sid_in_focus (str): sid for which the index is calculated
        volumeTable (pandas dataframe object): dataframe with relevant values for computing WPMI
        sid_count (int): # of total sids against which WPMI is computed
        sid_list (list): list of all sids against which WPMI is computed
    
    Returns:
        wpmi_list (numpy array): array containing WPMI values for the input sid_in_focus
    '''
    wpmi_list = np.empty(sid_count)
    sid_in_focus_arr = np.array(weightPerMonthTable.loc[sid_in_focus])
    comparison_arr = np.array(weightPerMonthTable)
    for x in range(sid_count):
        if(sid_list[x]!=sid_in_focus):
            wpmi_numerator = np.minimum(sid_in_focus_arr,comparison_arr[x]).sum()
            wpmi_denominator = np.maximum(sid_in_focus_arr,comparison_arr[x]).sum()
            wpmi_list[x] = wpmi_numerator/wpmi_denominator
        else:
            wpmi_list[x] = -1  # return a value of -1 when comparing a customer to themselves        
    return wpmi_list

def calculate_volume_per_zone_index(sid_in_focus, volumePerZoneTable, sid_count, sid_list):
    '''
    Creates the volume per zone index(VPZI). Index 5 of 6 for calculating Similarity Scores
    
    Args:
        sid_in_focus (str): sid for which the index is calculated
        volumeTable (pandas dataframe object): dataframe with relevant values for computing VPZI
        sid_count (int): # of total sids against which VPZI is computed
        sid_list (list): list of all sids against which VPZI is computed
    
    Returns:
        vpzi_list (numpy array): array containing VPZI values for the input sid_in_focus
    '''
    vpzi_list = np.empty(sid_count)
    sid_in_focus_arr = np.array(volumePerZoneTable.loc[sid_in_focus])
    comparison_arr = np.array(volumePerZoneTable)
    for x in range(sid_count):
        if(sid_list[x]!=sid_in_focus):
            vpzi_numerator = np.minimum(sid_in_focus_arr,comparison_arr[x]).sum()
            vpzi_denominator = np.maximum(sid_in_focus_arr,comparison_arr[x]).sum()
            vpzi_list[x] = vpzi_numerator/vpzi_denominator
        else:
            vpzi_list[x] = -1  # return a value of -1 when comparing a customer to themselves        
    return vpzi_list

def calculate_weight_per_zone_index(sid_in_focus, weightPerZoneTable, sid_count, sid_list):
    '''
    Creates the weight per zone index(WPZI). Index 6 of 6 for calculating Similarity Scores
    
    Args:
        sid_in_focus (str): sid for which the index is calculated
        volumeTable (pandas dataframe object): dataframe with relevant values for computing WPZI
        sid_count (int): # of total sids against which WPZI is computed
        sid_list (list): list of all sids against which WPZI is computed
    
    Returns:
        wpzi_list (numpy array): array containing WPZI values for the input sid_in_focus
    '''
    wpzi_list = np.empty(sid_count)
    sid_in_focus_arr = np.array(weightPerZoneTable.loc[sid_in_focus])
    comparison_arr = np.array(weightPerZoneTable)
    for x in range(sid_count):
        if(sid_list[x]!=sid_in_focus):
            wpzi_numerator = np.minimum(sid_in_focus_arr,comparison_arr[x]).sum()
            wpzi_denominator = np.maximum(sid_in_focus_arr,comparison_arr[x]).sum()
            wpzi_list[x] = wpzi_numerator/wpzi_denominator
        else:
            wpzi_list[x] = -1  # return a value of -1 when comparing a customer to themselves        
    return wpzi_list
####################################################################################################################    
#Calculate Similarity Scores

default_weights = np.full(6,1/6)
def get_weights(input_weights):
    '''
    Generate an array of weights for each index
    
    Args:
        input_weights (dict): user-provided weights for each index

    Returns:
        weight_arr (numpy array): array containing weights in the necessary order required for computing
        Similarity Scores
    '''
    weight_arr = np.zeros(6)
    weight_arr[0]=input_weights['weight_vs']
    weight_arr[1]=input_weights['weight_ws']
    weight_arr[2]=input_weights['weight_vpm']
    weight_arr[3]=input_weights['weight_wpm']
    weight_arr[4]=input_weights['weight_vpz']
    weight_arr[5]=input_weights['weight_wpz']
    return weight_arr
    
def get_similarity_score_params(arg_df):
    '''
    Generate all parameters needed as input for computing similarity scores
    
    Args:
        df (pandas dataframe object): Input dataframe. This will contain all relevant benchmarking attributes

    Returns:
        allPivots (list): list of all 6 pivot tables needed for computing the 6 Similarity Score indices
        sid_count (int): Count of sids in the input dataframe
        sid_list (list): list of all sids for which Similarity Scores will be computed. 
                         These become the index/column of the final Similarity Score matrix. 
    '''
    benchmarking_df = get_relevant_features(arg_df)
    allPivots = [x for x in create_PivotTables(benchmarking_df)]
    sid_list = allPivots[0].index.values
    sid_count = len(sid_list)
    return allPivots, sid_count, sid_list

def calculate_similarity_score(sid_in_focus, params, weights = default_weights):
    '''
    Calculate the similarity score of a selected sid
    
    Args:
        sid_in_focus (str): sid for which the score is calculated
        params (list): list object containing the parameters returned from get_similarity_score_params()
        weights (numpy array): weights for each of the 6 indices. Has a default state of np.full(6,1/6)

    Returns:
        ss_array (numpy array): An array containing Similarity Scores for the sid_in_focus vs all compared sids 
    '''
    sid_cnt = params[1]
    sid_ls = params[2]
    vsiList = calculate_volumetric_scale_index(sid_in_focus, volumeTable= params[0][0], sid_count= sid_cnt, sid_list=sid_ls)
    wsiList = calculate_weight_scale_index(sid_in_focus, weightTable= params[0][1], sid_count= sid_cnt, sid_list=sid_ls)
    vpmiList = calculate_volume_per_month_index(sid_in_focus, volumePerMonthTable= params[0][2], sid_count= sid_cnt, sid_list=sid_ls)
    wpmiList = calculate_weight_per_month_index(sid_in_focus, weightPerMonthTable= params[0][3], sid_count= sid_cnt, sid_list=sid_ls)
    vpziList = calculate_volume_per_zone_index(sid_in_focus, volumePerZoneTable= params[0][4], sid_count= sid_cnt, sid_list=sid_ls)
    wpziList = calculate_weight_per_zone_index(sid_in_focus, weightPerZoneTable= params[0][5], sid_count= sid_cnt, sid_list=sid_ls)
    ss_array = 100*(weights[0]*vsiList + weights[1]*wsiList + weights[2]*vpmiList + weights[3]*wpmiList + weights[4]*vpziList + weights[5]*wpziList)
    return np.round(ss_array,3)

#Function for creating a filename
def create_filename(artifact_type):
    '''
    Create filename given input string identifying file type
    
    Args:
        artifact_type (str): brief description of document being created, ie, "similarity_matrix" or "KPI_database"
        
    Returns:
        filename (str): filename with stipulated naming convention as follows:
            <artifact_type>_<timestamp in YYYYMMDD-HHMM format>.pkl.z
    '''
    timestamp = utilities.get_timestamp()
    output_path = os.path.join(orig_dir, paths.data_benchmarking_dir) 
    filename = output_path + "\\"+ artifact_type + "_" + timestamp + ".pkl.z"
    return filename
####################################################################################################################    
#Create full similarity score matrix
    
'''
#Keep for testing
#ref_sids = ['7B8E9E45F5','E0518749DD','924052DDF5','AACOENJW','FHVD7BDTDG']
def create_similarity_score_matrix(sid_list):
    
    similarity_score_table = pd.DataFrame(index = sid_list, columns = matrix_cols).fillna(-1)
    for x in pbar(sid_list):
        row_vals = calculate_similarity_score(sid_in_focus = x, params = params, weights=default_weights)
        if row_vals[0]==-1:
            break
        else:
            similarity_score_table.loc[x] = row_vals
    return similarity_score_table
    
params = get_similarity_score_params(raw_data)
matrix_cols = params[0][0].index.values
ref_sids = matrix_cols
ssmatrix = create_similarity_score_matrix(ref_sids)
'''

#for production
def create_similarity_score_matrix(extract_data, input_weights):
    '''
    Create similarity_score_matrix
    
    Args:
        extract_data (pandas dataframe object): raw dataframe from 71lbs database
       input_weights (numpy array): user-provided weights

    Returns:
        True (bool): No specific output. However, function creates and stores the matrix in output_dir
    '''
    params = get_similarity_score_params(extract_data)
    idx = params[0][0].index.values
    similarity_score_matrix = pd.DataFrame(index = idx, columns = idx).fillna(-1)
    matrix_weights = get_weights(input_weights)
    print("Calculating similarity score...")
    for x in tqdm(idx):
        row_vals = calculate_similarity_score(sid_in_focus = x, params = params, weights=matrix_weights)
        if row_vals[0]==-1:
            break
        else:
            similarity_score_matrix.loc[x] = row_vals

    print(f"\nSaving similarity score matrix...")
    filename = create_filename("similarity_score_matrix")
    joblib.dump(similarity_score_matrix, filename)
    print(f"Similarity score matrix stored in {filename}")
    return True

###########################################################################################
#After creating the similarity_score_matrix, we create the KPI database

#There are 2 steps here:
#1. Add standard methods
#2. Create descriptors to be used for similar sids
#3. Create KPI database

#Add standard methods

#Pass raw_data as an argument to this method
def add_standard_methods(arg_df):
    '''
    Method to add standard timeframes (as provided by Jose) as a new feature. This new feature will then be used to calculate KPIs by the 
    create_customer_KPI_database function.
    
    Args:
        arg_df (pandas dataframe object): raw dataframe from 71lbs database

    Returns:
        complete_df (pandas dataframe object): dataframe with standardized timeframes 
    
    '''
    arg_df = arg_df.reset_index()
    # Creating and filling the dictionary with the correct mappings of standardized windows/timeframes
    std_timeframes_dict = {}
    for method in arg_df.std_service_type.unique():
        if method == 'Ground':
            std_timeframes_dict['Ground'] = "Ground"
        elif method == 'Smartpost':
            std_timeframes_dict['Smartpost'] = "Ground"
        elif method == 'Home Delivery':
            std_timeframes_dict['Home Delivery'] = "Ground"
        elif method == '2Day':
            std_timeframes_dict['2Day'] = "2nd day EOD"
        elif method == 'Surepost':
            std_timeframes_dict['Surepost'] = "Ground"
        elif method == 'Priority Overnight':
            std_timeframes_dict['Priority Overnight'] = "Next Day 10.30 am"
        elif method == 'Express Saver':
            std_timeframes_dict['Express Saver'] = "3 day"
        elif method == '2nd Day Air':
            std_timeframes_dict['2nd Day Air'] = "2nd day EOD"
        elif method == 'Next Day Air':
            std_timeframes_dict['Next Day Air'] = "Next Day 3 pm"
        elif method == '2Day AM':
            std_timeframes_dict['2Day AM'] = "2nd day am"
        elif method == 'Standard Overnight':
            std_timeframes_dict['Standard Overnight'] = "Next Day 3 pm"
        elif method == '3 Day Select':
            std_timeframes_dict['3 Day Select'] = "3 day"
        elif method == 'First Overnight':
            std_timeframes_dict['First Overnight'] = "Next Day 10.30 am"
        elif method == 'Next Day Air Early':
            std_timeframes_dict['Next Day Air Early'] = "Next Day 8 am"
        elif method == '2nd Day Air A.M.':
            std_timeframes_dict['2nd Day Air A.M.'] = "2nd day am"
        elif method == 'Standard':
            std_timeframes_dict['Standard'] = "Ground"
        elif method == 'Next Day Air Saver':
            std_timeframes_dict['Next Day Air Saver'] = "Next Day 3 pm"
    
    new_df = np.array(arg_df['std_service_type'])
    # Array to hold standard windows/timeframes to be appended to df
    std_timeframes = []

    for method in new_df:
        std_timeframes.append(std_timeframes_dict[method])

    df = pd.DataFrame(std_timeframes)
    df.columns = ['std_timeframes']
    
    complete_df = pd.concat([arg_df, df], axis = 1)
    return complete_df
 
    
def get_descriptors(arg_df):
    """
    This function generates customer descriptors to be appended to final peer group outputs. The descriptors
    are made availabel for the purpose of reviewing the rationale relationship customers would have with
    one another. If customers serving the aerospace industry show up in the same peer group as customers
    serving consumer products, some adjustments to the benchmarking weights may be warranted.
    
    Args:
        dataframe of shipments containing the following columns at a minimum:
        ['business_sid', 'industry', 'sub_industry', 'sender_state', 'recipient_state']
    
    Returns: 
        dataframe of customers containing the following columns:
        ['business_sid', 'industry', 'sub_industry', 'pri_ship_origin_state', 'pri_ship_dest_state'].
        
    The 'pri_ship_origin_state' is the primary state from where parcels are shipped
    The 'pri_ship_dest_state' is the primary state to where parcels are shipped
    """
    
    def get_descriptor(arg_df, col):
        """
        Helper function to aggregate a specific descriptor to the customer level
        """
        arg_df = pd.DataFrame(arg_df.groupby(['business_sid', col])[col].nunique())
        arg_df.columns = ['count']
        arg_df = arg_df.sort_values(by='count', ascending=False)
        arg_df = arg_df.drop(columns='count')
        arg_df = arg_df.reset_index(level=col)
        arg_df = arg_df.loc[~arg_df.index.duplicated(keep='first')]

        return arg_df

    # preparing a unique list of business_sids to merge dataframes to
    descriptors_df = pd.DataFrame(arg_df.business_sid.unique(), columns=['business_sid'])
    descriptors_df = descriptors_df.set_index('business_sid')

    # builds dataframe with customer descriptors
    descriptors = ['industry', 'sub_industry', 'sender_state', 'recipient_state']
    for descriptor in descriptors:
        descriptor_df = get_descriptor(arg_df[['business_sid', descriptor]], descriptor)
        descriptors_df = pd.merge(descriptors_df, descriptor_df, how='outer', on='business_sid')

    # relabels columns
    descriptors_df.columns = ['industry', 'sub_industry', 'pri_ship_origin_state', 'pri_ship_dest_state']
    return descriptors_df


#Next, create the KPI database

def create_customer_KPI_database(arg_df):
    '''
    Function to create the dataframe to be imported into a database which would then be
    queried each time a similarity calculation is made.
    
    The 9 broad KPI Metrics are as follows:
        1. Average total spend
        2. Average total discounts 
        3. Average spend per shipping method 
        4. Average spend per month 
        5. Average discounts per shipping method 
        6. Proportion of shipping methods used 
        7. Total volume shipped per shipping method 
        8. Proportion of each carrier (FedEx, UPS) used 
        9. Average discounts per zone
        
    Each of these includes "sub-categories" of metrics, as defined below. Note that each of these 
    separate metrics add up to a total of 28 unique KPIs:
        1. Average total spend:
            a. Average total spend
            b. Average total spend per shipper
            c. Average total spend per lb 
            d. Average total spend per lb per shipper
        2. Average total discounts:
            a. Average total discount
            b. Average total discount per shipper
            c. Average total discount per lb 
            d. Average total discount per lb per shipper
        3. Average spend per shipping method:
            a. Average spend per method
            b. Average spend per method per lb
            c. Average spend per method per shipper
            d. Average spend per method per lb per shipper
        4. Average spend per month:
            a. Average spend per month
            b. Average spend per month per lb
            c. Average spend per month per shipper
            d. Average spend per month per lb per shipper
        5. Average discount per shipping method:
            a. Average discount per method
            b. Average discount per method per lb
            c. Average discount per method per shipper
            d. Average discount per method per lb per shipper
        6. Proportion of shipping methods used
            a. Average volume proportion per method
        7. Total volume shipper per shipping method
            a. Average volume per method
            b. Average volume per method per shipper
        8. Proportion of each carrier used
            a. Proportional carrier use(%FedEx, %UPS)
        9. Average discounts per zone
            a. Average discount per zone
            b. Average discount per zone per lb
            c. Average discount per zone per shipper
            d. Average discount per zone per lb per shipper 
            
        Args:
            args_df (pandas dataframe object): raw dataframe from 71lbs database
            
        Returns:
            return_df (pandas dataframe object): dataframe containing all KPIs
    '''
    print('Compiling Customer Metrics...')
    #Step 0.0: Create net discount column
    arg_df['net_discount_amount'] = np.add(abs(np.array(arg_df['freight_discount_amount'])),\
                            np.array(arg_df['misc_discount_amount'])  )
    #Step 0.1: Remove rows where weight is equal to 0 (otherwise, calculations would be undefined)
    arg_df = arg_df[arg_df['weight'] > 0]
    
    #Step 0.2: Add the new feature for standard method windows/timeframes as provided by Jose
    arg_df = add_standard_methods(arg_df)
    
    #Step 0.3: Add new feature for standard weight (pounds per package shipped)
    arg_df['std_weight'] = arg_df['weight'] / arg_df['package_count']
    
    #Step 0.4: Add new feature for average spend per lb of standard weight
    arg_df['avg_spend_lb'] = arg_df['net_charge_amount'] / arg_df['std_weight']
    
    #Step 0.5: Add new feature for average discounts per lb of standard weight
    arg_df['avg_disc_lb'] = arg_df['net_discount_amount'] / arg_df['std_weight']
    
    ################################################################
    #Begin KPI Calculations
    # 1.1 Average total spend
    avg_spend_df = arg_df[['business_sid', 'net_charge_amount']]
    avg_tot_spend = avg_spend_df.groupby(['business_sid']).mean()
    avg_tot_spend.columns = ['avg_tot_spend']
    
    # 1.2 Average total spend per shipper
    avg_spend_per_shipper = pd.pivot_table(arg_df, values = 'net_charge_amount',
                                           index = ['business_sid'],
                                           columns = 'shipper',
                                           aggfunc = np.mean, fill_value = 0)
    
    avg_spend_per_shipper.columns = ['avg_spend_fedex', 'avg_spend_ups']    
    
    # 1.3 Average total spend per pound
    avg_spend_per_lb_df = arg_df[['business_sid', 'avg_spend_lb']]
    avg_spend_per_lb = avg_spend_per_lb_df.groupby(['business_sid']).mean()
    avg_spend_per_lb.columns = ['avg_tot_spend_per_lb']
    
    # 1.4 Average total spend per pound, per shipper
    avg_spend_per_lb_shipper = pd.pivot_table(arg_df, values = 'avg_spend_lb',
                                           index = ['business_sid'],
                                           columns = 'shipper',
                                           aggfunc = np.mean, fill_value = 0)
    
    avg_spend_per_lb_shipper.columns = ['avg_spend_per_lb_fedex', 'avg_spend_per_lb_ups']  
    
    ################################################################
    # 2.1 Average total discounts
    avg_disc_df = arg_df[['business_sid', 'net_discount_amount']]
    avg_tot_discounts = abs(avg_disc_df.groupby(['business_sid']).mean())
    avg_tot_discounts.columns = ['avg_tot_discounts']
    
    # 2.2 Average total discounts per shipper
    avg_disc_per_shipper = pd.pivot_table(arg_df, values = 'net_discount_amount',
                                           index = ['business_sid'],
                                           columns = 'shipper',
                                           aggfunc = np.mean, fill_value = 0)
    
    avg_disc_per_shipper.columns = ['avg_disc_fedex', 'avg_disc_ups']
    
    # 2.3 Average total discounts per pound
    avg_disc_lb_df = arg_df[['business_sid', 'avg_disc_lb']]
    avg_disc_per_lb = avg_disc_lb_df.groupby(['business_sid']).mean()
    avg_disc_per_lb.columns = ['avg_tot_disc_per_lb']
    
    # 2.4 Average total discounts per pound, per shipper
    avg_disc_per_lb_shipper = pd.pivot_table(arg_df, values = 'avg_disc_lb',
                                           index = ['business_sid'],
                                           columns = 'shipper',
                                           aggfunc = np.mean, fill_value = 0)
    
    avg_disc_per_lb_shipper.columns = ['avg_disc_per_lb_fedex', 'avg_disc_per_lb_ups']  
    
    ################################################################
    # 3.1 Pivot table for average spend per method
    avg_spend_per_method_pt = pd.pivot_table(arg_df, values = 'net_charge_amount', 
                                            index = ['business_sid'], 
                                            columns = 'std_timeframes', 
                                            aggfunc = np.mean, fill_value = 0)
    
    avg_spend_per_method_pt.columns = ['avg_spend_Ground', 'avg_spend_2nd_day_EOD', 'avg_spend_Next_Day_10.30_am', 
                                       'avg_spend_3_day', 'avg_spend_Next_Day_3_pm', 'avg_spend_2nd_day_am', 
                                       'avg_spend_Next_Day_8_am']
    
    # 3.2 Average spend per method per lb
    avg_spend_per_method_per_lb_df = pd.pivot_table(arg_df, values = 'avg_spend_lb', 
                                            index = ['business_sid'], 
                                            columns = 'std_timeframes', 
                                            aggfunc = np.mean, fill_value = 0)
    per_lb_cols_spend = []
    for col in avg_spend_per_method_pt.columns:
        per_lb_cols_spend.append(str(col) + "_per_lb")
        
    avg_spend_per_method_per_lb_df.columns = per_lb_cols_spend
    
    # 3.3 Average spend per method per shipper
    avg_spend_per_method_shipper = pd.pivot_table(arg_df, values = 'net_charge_amount', 
                                            index = ['business_sid'], 
                                            columns = ['shipper', 'std_timeframes'], 
                                            aggfunc = np.mean, fill_value = 0)
        
    avg_spend_per_method_shipper.columns = ['avg_spend_2nd_day_eod_fedex', 'avg_spend_2nd_day_am_fedex', 'avg_spend_3_day_fedex', 
                                              'avg_spend_Ground_fedex', 'avg_spend_Next_day_10.30_am_fedex', 'avg_spend_Next_Day_3_pm_fedex',
                                              'avg_spend_2nd_day_eod_ups', 'avg_spend_2nd_day_am_ups', 'avg_spend_3_day_ups', 
                                              'avg_spend_Ground_ups', 'avg_spend_Next_Day_3_pm_ups', 'avg_spend_Next_day_8_am_ups']
    
    # 3.4 Average total spend per method, per pound, per shipper
    avg_spend_per_method_per_lb_per_shipper = pd.pivot_table(arg_df, values = 'avg_spend_lb',
                                           index = ['business_sid'],
                                           columns = ['shipper', 'std_timeframes'],
                                           aggfunc = np.mean, fill_value = 0)
    
    per_method_per_lb_per_shipper_cols = []
    for col in avg_spend_per_method_shipper.columns:
        per_method_per_lb_per_shipper_cols.append(str(col) + "_per_lb")
    
    avg_spend_per_method_per_lb_per_shipper.columns = per_method_per_lb_per_shipper_cols
    
    ################################################################
    # 4.1 Pivot table for average spend per month
    arg_df['month'] = arg_df.shipment_date.dt.month
    avg_spend_per_month_pt = pd.pivot_table(arg_df, values = 'net_charge_amount',
                                           index = ['business_sid'],
                                           columns = 'month',
                                           aggfunc = np.mean, fill_value = 0)
    
    avg_spend_per_month_pt.columns = ['avg_spend_Jan', 'avg_spend_Feb', 'avg_spend_Mar', 'avg_spend_Apr', 
                                      'avg_spend_May', 'avg_spend_Jun', 'avg_spend_Jul', 'avg_spend_Aug', 
                                      'avg_spend_Sep', 'avg_spend_Oct', 'avg_spend_Nov', 'avg_spend_Dec']
    
    # 4.2 Average spend per month per shipper
    avg_spend_per_month_per_shipper_df = pd.pivot_table(arg_df, values = 'net_charge_amount',
                                           index = ['business_sid'],
                                           columns = ['shipper', 'month'],
                                           aggfunc = np.mean, fill_value = 0)
    per_month_per_shipper_cols = []
    for col in avg_spend_per_month_pt.columns:
        per_month_per_shipper_cols.append(str(col) + "_fedex")
        
    for col in avg_spend_per_month_pt.columns:
        per_month_per_shipper_cols.append(str(col) + "_ups")
        
    avg_spend_per_month_per_shipper_df.columns = per_month_per_shipper_cols
    
    # 4.3 Average spend per month per lb
    avg_spend_per_month_per_lb_df = pd.pivot_table(arg_df, values = 'avg_spend_lb', 
                                            index = ['business_sid'], 
                                            columns = 'month', 
                                            aggfunc = np.mean, fill_value = 0)
    month_per_lb_cols_spend = []
    for col in avg_spend_per_month_pt.columns:
        month_per_lb_cols_spend.append(str(col) + "_per_lb")
        
    avg_spend_per_month_per_lb_df.columns = month_per_lb_cols_spend
    
    # 4.4 Average spend per month, per lb, per shipper
    avg_spend_per_month_per_lb_per_shipper = pd.pivot_table(arg_df, values = 'avg_spend_lb',
                                           index = ['business_sid'],
                                           columns = ['shipper', 'month'],
                                           aggfunc = np.mean, fill_value = 0)
    
    per_month_per_lb_per_shipper_cols = []
    for col in per_month_per_shipper_cols:
        per_month_per_lb_per_shipper_cols.append(str(col) + "_per_lb")
    
    avg_spend_per_month_per_lb_per_shipper.columns = per_month_per_lb_per_shipper_cols
    
    ################################################################
    # 5.1 Pivot table for average discount per method
    avg_disc_per_method_pt = abs(pd.pivot_table(arg_df, values = 'net_discount_amount',
                                           index = ['business_sid'],
                                           columns = 'std_timeframes',
                                           aggfunc = np.mean, fill_value = 0))
    
    avg_disc_per_method_pt.columns = ['avg_disc_Ground', 'avg_disc_2nd_day_EOD', 'avg_disc_Next_Day_10.30_am', 
                                       'avg_disc_3_day', 'avg_disc_Next_Day_3_pm', 'avg_disc_2nd_day_am', 
                                       'avg_disc_Next_Day_8_am']
    
    # 5.2 Average discount per method, per lb
    avg_disc_per_method_per_lb_df = pd.pivot_table(arg_df, values = 'avg_disc_lb', 
                                            index = ['business_sid'], 
                                            columns = 'std_timeframes', 
                                            aggfunc = np.mean, fill_value = 0)
    per_lb_cols_disc = []
    for col in avg_disc_per_method_pt.columns:
        per_lb_cols_disc.append(str(col) + "_per_lb")
        
    avg_disc_per_method_per_lb_df.columns = per_lb_cols_disc
    
    # 5.3 Average disc per method per shipper
    avg_disc_per_method_shipper = pd.pivot_table(arg_df, values = 'net_discount_amount', 
                                            index = ['business_sid'], 
                                            columns = ['shipper', 'std_timeframes'], 
                                            aggfunc = np.mean, fill_value = 0)
        
    avg_disc_per_method_shipper.columns = ['avg_disc_2nd_day_eod_fedex', 'avg_disc_2nd_day_am_fedex', 'avg_disc_3_day_fedex', 
                                              'avg_disc_Ground_fedex', 'avg_disc_Next_day_10.30_am_fedex', 'avg_disc_Next_Day_3_pm_fedex',
                                              'avg_disc_2nd_day_eod_ups', 'avg_disc_2nd_day_am_ups', 'avg_disc_3_day_ups', 
                                              'avg_disc_Ground_ups', 'avg_disc_Next_Day_3_pm_ups', 'avg_disc_Next_day_8_am_ups']
    
    
    # 5.4 Average total discount per method, per pound, per shipper
    avg_disc_per_method_per_lb_per_shipper = pd.pivot_table(arg_df, values = 'avg_disc_lb',
                                           index = ['business_sid'],
                                           columns = ['shipper', 'std_timeframes'],
                                           aggfunc = np.mean, fill_value = 0)
    
    per_method_per_lb_per_shipper_disc_cols = []
    for col in avg_disc_per_method_shipper.columns:
        per_method_per_lb_per_shipper_disc_cols.append(str(col) + "_per_lb")
    
    avg_disc_per_method_per_lb_per_shipper.columns = per_method_per_lb_per_shipper_cols
           
    ################################################################
    # 6. Proportions of shipping methods used
    prop_methods_df = arg_df[['business_sid', 'weight', 'std_timeframes']]
    tempvoldf = prop_methods_df.groupby(['business_sid', 'std_timeframes']).count()
    tempvoldf.columns = ['Count']
    volume_count_method_pt = pd.pivot_table(tempvoldf, values = 'Count', index = ['business_sid'], columns = 'std_timeframes', 
                     aggfunc = np.sum, fill_value = 0)
    
    volume_proportion_per_method = volume_count_method_pt.apply(lambda x: x/x.sum(), axis = 1)
    
    volume_proportion_per_method.columns = ['vp_Ground', 'vp_2nd_day_EOD', 'vp_Next_Day_10.30_am', 
                                       'vp_3_day', 'vp_Next_Day_3_pm', 'vp_2nd_day_am', 
                                       'vp_Next_Day_8_am']
    
    ################################################################
    # 7.1 Total volume per method used
    tot_vol_method_df = arg_df[['business_sid', 'shipper', 'std_timeframes']]
    tempvol_df = tot_vol_method_df.groupby(['business_sid', 'std_timeframes']).count()
    tempvol_df.columns = ['Count']

    volume_count_method_pt = pd.pivot_table(tempvol_df, values = 'Count', 
                                            index = ['business_sid'], 
                                            columns = 'std_timeframes', 
                                            aggfunc = np.sum, fill_value = 0)

    volume_count_method_pt.columns = ['vc_Ground', 'vc_2nd_day_EOD', 'vc_Next_Day_10.30_am', 
                                       'vc_3_day', 'vc_Next_Day_3_pm', 'vc_2nd_day_am', 
                                       'vc_Next_Day_8_am']
    
    # 7.2 Total volume per method used per shipper
    arg_df['shipper_std_timeframes'] = arg_df['shipper'] + arg_df['std_timeframes']
    tot_vol_method_shipper_df = arg_df[['business_sid', 'shipper', 'shipper_std_timeframes']]
    tempvol_shipper_df = tot_vol_method_shipper_df.groupby(['business_sid', 'shipper_std_timeframes']).count()
    tempvol_shipper_df.columns = ['Count']
    volume_count_method_shipper_pt = pd.pivot_table(tempvol_shipper_df, values = 'Count', 
                                            index = ['business_sid'], 
                                            columns = ['shipper_std_timeframes'], 
                                            aggfunc = np.sum, fill_value = 0)

    volume_count_method_shipper_pt.columns = ['2nd_day_eod_fedex_vc', '2nd_day_am_fedex_vc', '3_day_fedex_vc', 
                                              'Ground_fedex_vc', 'Next_day_10.30_am_fedex_vc', 'Next_Day_3_pm_fedex_vc',
                                              '2nd_day_eod_ups_vc', '2nd_day_am_ups_vc', '3_day_ups_vc', 
                                              'Ground_ups_vc', 'Next_Day_3_pm_ups_vc', 'Next_day_8_am_ups_vc']
    
    ################################################################
    # 8. Proportional carrier use (fedex/ups)
    prop_shipper_df = arg_df[['business_sid', 'weight', 'shipper']]
    tempvoldf = prop_shipper_df.groupby(['business_sid', 'shipper']).count()
    tempvoldf.columns = ['Count']
    
    prop_count_method_pt = pd.pivot_table(tempvoldf, values = 'Count', index = ['business_sid'], columns = 'shipper', 
                     aggfunc = np.sum, fill_value = 0)

    proportion_per_shipper = prop_count_method_pt.apply(lambda x: x/x.sum(), axis = 1)
    proportion_per_shipper.columns = ['proportion_fedex', 'proportion_ups']
    ################################################################
    # 9.1 Pivot table for average discount per zone
    avg_disc_per_zone_pt = abs(pd.pivot_table(arg_df, values = 'net_discount_amount',
                                           index = ['business_sid'],
                                           columns = 'zone',
                                           aggfunc = np.mean, fill_value = 0))
    
    avg_disc_per_zone_pt_columns = []
    for zone in avg_disc_per_zone_pt.columns:
        avg_disc_per_zone_pt_columns.append("avg_disc_zone_" + str(zone))
    
    avg_disc_per_zone_pt.columns = avg_disc_per_zone_pt_columns
    
    # 9.2 Average discount per zone per lb
    avg_disc_per_zone_per_lb_df = pd.pivot_table(arg_df, values = 'avg_disc_lb', 
                                            index = ['business_sid'], 
                                            columns = 'zone', 
                                            aggfunc = np.mean, fill_value = 0)
    zone_per_lb_cols_disc = []
    for col in avg_disc_per_zone_pt.columns:
        zone_per_lb_cols_disc.append(str(col) + "_per_lb")
        
    avg_disc_per_zone_per_lb_df.columns = zone_per_lb_cols_disc
    
    # 9.3 Average discount per zone per shipper
    avg_disc_per_zone_shipper = pd.pivot_table(arg_df, values = 'net_discount_amount', 
                                            index = ['business_sid'], 
                                            columns = ['shipper', 'zone'], 
                                            aggfunc = np.mean, fill_value = 0)
        
    zone_per_shipper_cols_disc = []
    for col in avg_disc_per_zone_shipper.columns:
        zone_per_shipper_cols_disc.append("avg_disc_zone_"+str(col[1]) +"_"+ str(col[0]))
    
    avg_disc_per_zone_shipper.columns = zone_per_shipper_cols_disc
    
    # 9.4 Average discount per zone, per lb, per shipper
    avg_disc_per_zone_per_lb_per_shipper = pd.pivot_table(arg_df, values = 'avg_disc_lb',
                                           index = ['business_sid'],
                                           columns = ['shipper', 'zone'],
                                           aggfunc = np.mean, fill_value = 0)
    
    per_zone_per_lb_per_shipper_disc_cols = []
    for col in avg_disc_per_zone_per_lb_per_shipper.columns:
        per_zone_per_lb_per_shipper_disc_cols.append("avg_disc_zone_"+str(col[1]) +"_"+ str(col[0]) + "_per_lb")
    
    avg_disc_per_zone_per_lb_per_shipper.columns = per_zone_per_lb_per_shipper_disc_cols
    ################################################################
    #Also add descriptors to this table. This will be used for collectiing data on similar SIDs
    descriptor_df = get_descriptors(arg_df)
    # Concatenate all dataframes into final product
    return_df = pd.concat(
            [avg_tot_spend,
             avg_spend_per_shipper,
             avg_spend_per_lb,
             avg_spend_per_lb_shipper,
             avg_spend_per_method_pt,
             avg_spend_per_method_per_lb_df,
             avg_spend_per_method_shipper,
             avg_spend_per_method_per_lb_per_shipper,
             avg_spend_per_month_pt,
             avg_spend_per_month_per_lb_df,
             avg_spend_per_month_per_shipper_df,
             avg_spend_per_month_per_lb_per_shipper,
             avg_tot_discounts,
             avg_disc_per_shipper,
             avg_disc_per_lb,
             avg_disc_per_lb_shipper,
             avg_disc_per_method_pt,
             avg_disc_per_method_per_lb_df,
             avg_disc_per_method_shipper, 
             avg_disc_per_method_per_lb_per_shipper,
             volume_proportion_per_method,
             volume_count_method_pt,
             volume_count_method_shipper_pt,
             proportion_per_shipper,
             descriptor_df,
             avg_disc_per_zone_pt,
             avg_disc_per_zone_per_lb_df,
             avg_disc_per_zone_shipper, 
             avg_disc_per_zone_per_lb_per_shipper], axis=1)
    
    filename = create_filename("KPI_database")
    print("Saving KPI database...")
    joblib.dump(return_df, filename)
    print(f"KPI database stored in {filename}\n")
    return return_df

#for i,j in enumerate(a.columns):
 #   print(i, ':', j)

##########################################################################################


