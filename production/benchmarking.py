import datetime
import glob
import sys
import joblib
import zipcodes
import numpy as np
import pandas as pd
import time
from prettytable import PrettyTable
import delivery_prediction_predict
import paths
import menu_options

# Retrieve raw scores from the full SS matrix
def get_raw_scores(sid, preloaded_matrix, preloaded_KPIs):
    full_ss_matrix_optimized = np.array(preloaded_matrix)
    full_matrix_indices = np.array(preloaded_matrix.index.values)
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    # Retrieve the index at which the requested sid is present in the SS matrix
    test_sid_idx = np.where(full_matrix_indices == sid)[0][0]
    # Return the SS array
    raw_score_array = full_ss_matrix_optimized[test_sid_idx]
    # Change NANs to 0
    raw_score_array[np.isnan(raw_score_array)] = 0
    # Identify indices where scores <=0 are currently present
    dirty_score_idx = np.where(raw_score_array <= 0)
    # print(dirty_score_idx)
    # Update these indices with a different value (0.0001 is chosen here)
    clean_scores = raw_score_array
    # print(clean_scores[0:5])
    np.put(clean_scores, dirty_score_idx, np.full(shape=len(dirty_score_idx), fill_value=0.0001))
    # print(clean_scores[0:5])
    return np.array(clean_scores)


# For a given sid and percentile, this function returns the threshold similarity score value.
# All sids with a similarity score higher than the threshold value will be deemed 'similar customers'
# at the given percentile(default 90th %ile) level
def ss_threshold(sid, percentile, preloaded_matrix, preloaded_KPIs):
    full_ss_matrix_optimized = np.array(preloaded_matrix)
    full_matrix_indices = np.array(preloaded_matrix.index.values)
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    normalized_threshold_score = (np.percentile(get_raw_scores(sid, preloaded_matrix, preloaded_KPIs), percentile))
    # print(normalized_threshold_score)
    return np.round(normalized_threshold_score, 3)

# This directly returns a list of ALL similar customers that are above a certain percentile threshold(default 90th %ile)
# (Option 1)
def get_similar_customers(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    full_ss_matrix_optimized = np.array(preloaded_matrix)
    full_matrix_indices = np.array(preloaded_matrix.index.values)
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    inp_sid = input_sid
    # Get threshold score for requested sid
    threshold_score = ss_threshold(inp_sid, percentile, preloaded_matrix, preloaded_KPIs)
    # Get index of requested sid in the full_ss_matrix
    sid_idx = np.where(full_matrix_indices == inp_sid)[0][0]
    # Get raw scores for the test_sid from the full_ss_matrix
    sid_scores = get_raw_scores(full_matrix_indices[sid_idx], preloaded_matrix, preloaded_KPIs)
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
        print(inp_sid, 'has', len(sorted_sids), f'similar customers in the top {100 - percentile} percentile.\n')
    cols = ["Similar_SIDs", "Similarity_Score"]
    KPIdata = np.column_stack((sorted_sids, sorted_scores))
    df = pd.DataFrame(data=KPIdata, columns=cols)
    return sorted_sids, df


# Average Spend KPI (Option 2)
def KPI_avg_spend(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    full_ss_matrix_optimized = np.array(preloaded_matrix)
    full_matrix_indices = np.array(preloaded_matrix.index.values)
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    avg_spend_KPI_list = np.array(np.round(preloaded_KPIs['avg_tot_spend'], 2))

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs)[0]
    similar_customers_idx = np.isin(full_KPI_indices, similar_customer_list)
    self_avg_spend = np.round(preloaded_KPIs.loc[input_sid][0], 2)
    peer_avg_spend = np.round(np.mean(avg_spend_KPI_list[similar_customers_idx]), 2)
    if (user == 1):
        print("--- Average Spend ---")
        print(" ")
        print(f'Your average spend is: ${self_avg_spend}\nYour peers\' average spend is: ${peer_avg_spend}\n')
        print("----------------------------------------------------------")
        print(" ")
    # create result dataframe
    cols = ["Self", "Peers"]
    KPIdata = np.column_stack((self_avg_spend, peer_avg_spend))
    df = pd.DataFrame(data=KPIdata, columns=cols, index=['Average_Spend'])
    return df


# Average Discount KPI (Option 3)
def KPI_avg_discount(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    full_ss_matrix_optimized = np.array(preloaded_matrix)
    full_matrix_indices = np.array(preloaded_matrix.index.values)
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    avg_discount_KPI_list = np.array(np.round(preloaded_KPIs['avg_tot_discounts'], 2))

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs)[0]
    similar_customers_idx = np.isin(full_KPI_indices, similar_customer_list)

    self_avg_discount = np.round(preloaded_KPIs.loc[input_sid][0], 2)
    peer_avg_discount = np.round(np.mean(avg_discount_KPI_list[similar_customers_idx]), 2)
    if (user == 1):
        print("--- Average Discount ---")
        print(" ")
        print(
            f'Your average discount is: ${self_avg_discount}\nYour peers\' average discount is: ${peer_avg_discount}\n')
        print("----------------------------------------------------------")
        print(" ")
    # create result dataframe
    cols = ["Self", "Peers"]
    KPIdata = np.column_stack((self_avg_discount, peer_avg_discount))
    df = pd.DataFrame(data=KPIdata, columns=cols, index=['Average_Discount'])
    return df


# Average Spend per Shipping Method KPI (Option 4)
def KPI_avg_spend_per_method(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    full_ss_matrix_optimized = np.array(preloaded_matrix)
    full_matrix_indices = np.array(preloaded_matrix.index.values)
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    spend_per_method_df = np.round(preloaded_KPIs.iloc[:, 2:9], 2)

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs)[0]
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


# Average Discount per Shipping Method KPI (Option 5)
def KPI_avg_discount_per_method(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    full_ss_matrix_optimized = np.array(preloaded_matrix)
    full_matrix_indices = np.array(preloaded_matrix.index.values)
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    discount_per_method_df = np.round(preloaded_KPIs.iloc[:, 21:28], 2)

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs)[0]
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


# Monthly Spend KPI (Option 6)
def KPI_spend_per_month(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    full_ss_matrix_optimized = np.array(preloaded_matrix)
    full_matrix_indices = np.array(preloaded_matrix.index.values)
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    spend_per_month_df = np.round(preloaded_KPIs.iloc[:, 9:21], 2)
    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs)[0]
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
        print("--- Average Monthly Spend ---")
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


# Discounts per Zone KPI (Option 7)
def KPI_discount_per_zone(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    full_ss_matrix_optimized = np.array(preloaded_matrix)
    full_matrix_indices = np.array(preloaded_matrix.index.values)
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    discount_per_zone_df = np.round(preloaded_KPIs.iloc[:, 28:38], 2)

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs)[0]
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


# Shipping Method Proportion KPI (Option 8)
def KPI_methodwise_proportion(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    full_ss_matrix_optimized = np.array(preloaded_matrix)
    full_matrix_indices = np.array(preloaded_matrix.index.values)
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    methodwise_proportion_df = np.round(preloaded_KPIs.iloc[:, 38:45], 2)

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs)[0]
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


# Volume per method KPI (Option 9)
def KPI_volume_per_method(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    full_ss_matrix_optimized = np.array(preloaded_matrix)
    full_matrix_indices = np.array(preloaded_matrix.index.values)
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    volume_per_method_df = np.round(preloaded_KPIs.iloc[:, 45:-2], 2)

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs)[0]
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


# Shipper Proportion KPI (Option 10)
def KPI_shipper_proportion(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs):
    full_ss_matrix_optimized = np.array(preloaded_matrix)
    full_matrix_indices = np.array(preloaded_matrix.index.values)
    full_KPI_indices = np.array(preloaded_KPIs.index.values)
    fedex_prop_list = np.array(np.round(preloaded_KPIs['prop_fedex'], 2))
    ups_prop_list = np.array(np.round(preloaded_KPIs['prop_ups'], 2))

    similar_customer_list = get_similar_customers(input_sid, percentile, user, preloaded_matrix, preloaded_KPIs)[0]
    similar_customers_idx = np.isin(full_KPI_indices, similar_customer_list)

    self_fedex_proportion = np.round(preloaded_KPIs.loc[input_sid][-2], 2) * 100
    self_ups_proportion = np.round(preloaded_KPIs.loc[input_sid][-1], 2) * 100

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

def get_selected_metrics(selected_metrics, sid, percentile, preloaded_matrix, preloaded_KPIs):
    # if "Number of peers" in selected_metrics:
    #     KPI1 = get_similar_customers(sid, percentile, user=1)
    dispatch_list = []
    for metric in menu_options.benchmarking_metric_options:
        # If metric is selected: add function that corresponds to metric to dispatch list
        if metric in selected_metrics:
            dispatch_list.append(metric_to_function[metric](sid, percentile, user=1,
                                                            preloaded_matrix=preloaded_matrix,
                                                            preloaded_KPIs=preloaded_KPIs))

    # allKPIs = pd.concat([KPI1, KPI2, KPI3, KPI4, KPI5, KPI6, KPI7, KPI8, KPI9])
    allKPIs = pd.concat(dispatch_list)

    # Get a list of similar sids and respective similarity scores
    similarity_data = get_similar_customers(sid, percentile, 0, preloaded_matrix, preloaded_KPIs)[1]
    # print(allKPIs)
    # filename convention: benchmark_<SID>_<PERCENTILE>_<YYYYMMDD-HHMM>
    output_path = paths.output_benchmarking_dir
    timestr = time.strftime("%Y%m%d-%H%M")
    # print(output_path)
    file_name = output_path + '\Benchmarking_' + str(sid) + '_' + str(
        percentile) + '_' + timestr + '.xlsx'
    # print(file_name)

    with pd.ExcelWriter(file_name) as writer:
        allKPIs.to_excel(writer, sheet_name='KPIs')
        similarity_data.to_excel(writer, sheet_name='Reference_Data')
    return True



