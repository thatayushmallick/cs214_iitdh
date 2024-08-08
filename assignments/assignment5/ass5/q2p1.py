import pandas as pd
import numpy as np
from math import log2

# Data loading step
data_location = 'drug200.csv'
data_set = pd.read_csv(data_location)

def compute_entropy(sections, target_classes):
   total_count = float(sum([len(sect) for sect in sections]))
   entropy_result = 0.0
   for sect in sections:
       sect_length = float(len(sect))
       if sect_length == 0:
           continue
       entropy_sum = 0.0
       for target_class in target_classes:
           proportion = [element[-1] for element in sect].count(target_class) / sect_length
           entropy_sum -= proportion * log2(proportion) if proportion else 0
       entropy_result += (entropy_sum * sect_length / total_count)
   return entropy_result

def info_gain_evaluation(sections, target_classes, parent_entropy):
   total_sample = float(sum([len(sect) for sect in sections]))
   entropy_weighted = 0.0
   for sect in sections:
       length_sect = len(sect)
       if length_sect == 0:
           continue
       entropy_weighted += (length_sect / total_sample) * compute_entropy([sect], target_classes)
   return parent_entropy - entropy_weighted

def partition_data(feature_index, threshold, dataset):
   group_left, group_right = [], []
   for row in dataset:
       if row[feature_index] < threshold:
           group_left.append(row)
       else:
           group_right.append(row)
   return group_left, group_right

def optimal_split(data):
    outcomes = set(row[-1] for row in data)
    optimal_index, optimal_threshold, highest_gain, optimal_partitions = None, None, float('-inf'), None
    root_entropy = compute_entropy([data], outcomes)
    for idx in range(len(data[0])-1):
        for entry in data:
            current_partitions = partition_data(idx, entry[idx], data)
            current_gain = info_gain_evaluation(current_partitions, outcomes, root_entropy)
            if current_gain > highest_gain:
                optimal_index, optimal_threshold, highest_gain, optimal_partitions = idx, entry[idx], current_gain, current_partitions
    return {'feature': optimal_index, 'threshold': optimal_threshold, 'partitions': optimal_partitions}

def assign_leaf(nodes):
   result = [record[-1] for record in nodes]
   return max(set(result), key=result.count)

def node_split(node, depth_cap, smallest_size, current_depth):
    left_branch, right_branch = node['partitions']
    del(node['partitions'])

    if not left_branch or not right_branch:
        leaf_value = assign_leaf(left_branch + right_branch)
        node['left_branch'] = node['right_branch'] = leaf_value
        return

    if current_depth >= depth_cap:
        node['left_branch'], node['right_branch'] = assign_leaf(left_branch), assign_leaf(right_branch)
        return

    if len(left_branch) <= smallest_size:
        node['left_branch'] = assign_leaf(left_branch)
    else:
        node['left_branch'] = optimal_split(left_branch)
        node_split(node['left_branch'], depth_cap, smallest_size, current_depth + 1)

    if len(right_branch) <= smallest_size:
        node['right_branch'] = assign_leaf(right_branch)
    else:
        node['right_branch'] = optimal_split(right_branch)
        node_split(node['right_branch'], depth_cap, smallest_size, current_depth + 1)

def initiate_decision_tree(train_dataset, depth_threshold, size_limit):
   root_node = optimal_split(train_dataset)
   node_split(root_node, depth_threshold, size_limit, 1)
   return root_node

def classify(sample, node):
    while isinstance(node, dict):
        if sample[node['feature']] < node['threshold']:
            node = node['left_branch']
        else:
            node = node['right_branch']
    return node

def adapt_data(df):
    df['Age_group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=[0, 1, 2]).astype(int)
    df['Gender_code'] = pd.Categorical(df['Sex'], ['F', 'M']).codes
    df['BP_code'] = pd.Categorical(df['BP'], ['LOW', 'NORMAL', 'HIGH']).codes + 1
    df['Cholesterol_code'] = pd.Categorical(df['Cholesterol'], ['NORMAL', 'HIGH']).codes
    df['Medication_code'] = pd.Categorical(df['Drug'], ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']).codes
    return df

# Process the dataset and construct the decision tree
raw_data = pd.read_csv('drug200.csv')
df=adapt_data(data_set)
tar=df['Drug']
drug_mapping = {'drugA': '0', 'drugC': '2', 'drugX': '3', 'drugB': '1', 'drugY': '4'}
tar = tar.map(drug_mapping).fillna('unknown')
tar = pd.to_numeric(tar, errors='coerce')
tar=tar.values



processed_dataset = adapt_data(data_set).values.tolist()
decision_root = initiate_decision_tree(processed_dataset, 5, 10)

# Predictions and evaluation
medication_dict = {0: 'drugA', 1: 'drugB', 2: 'drugC', 3: 'drugX', 4: 'drugY'}
eval_predictions = [classify(record, decision_root) for record in processed_dataset]


right=0
for g in range(len(eval_predictions)):
    if(eval_predictions[g]==tar[g]):
        right+=1

print('Accuracy:',right/len(eval_predictions))

prediction_summary = {drug: 0 for drug in medication_dict.values()}
for prediction in eval_predictions:
    prediction_summary[medication_dict[prediction]] += 1

most_frequent_med = max(prediction_summary, key=prediction_summary.get)
print("Most frequently predicted medication:", most_frequent_med)
