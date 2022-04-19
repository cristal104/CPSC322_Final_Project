import csv
import math
import numpy as np
import random

# general util funtions 
def read_csv_to_table(filename):
    """reads a csv file and stores in a 2D list

    Args:
        filename(obj): name of csv file
    Returns:
        header(list of obj): list containing header
        table(list of list of obj): 2D list extracted from csv file

    Notes:
        Raise ValueError on invalid col_identifier
    """
    table = []
    with open(filename, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        for row in reader:
            table.append(row)
    return header, table

def get_column(table, header, col_name):
    """Extracts a column from the table data as a list.

    Args:
        col_identifier(str or int): string for a column name or int
            for a column index
        include_missing_values(bool): True if missing values ("NA")
            should be included in the column, False otherwise.

    Returns:
        list of obj: 1D list of values in the column

    Notes:
        Raise ValueError on invalid col_identifier
    """
    col_index = header.index(col_name)
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col

def change_isFraud(table, header, col_name):
    col_index = header.index(col_name)
    for row in table:
        value = row[col_index]
        if value == '0':
            row[col_index] = "no"
        if value == '1':
            row[col_index] = "yes"

def dictionary_to_list(dictionary):
    x_list = list(dictionary.keys())
    y_list = list(dictionary.values())
    return x_list, y_list

def get_frequencies(table, header, col_name):
    col = get_column(table, header, col_name)
    col.sort() # inplace 
    # parallel lists
    values = []
    counts = []
    for value in col:
        if value in values: # seen it before
            counts[-1] += 1 # okay because sorted
        else: # haven't seen it before
            values.append(value)
            counts.append(1)
    return values, counts

def drop_cols(table, header, col_name):
    col_index = header.index(col_name)
    for col in table:
        del col[col_index]
    del header[col_index]

def create_list(X_train, classes, key_value, total_classes, inx):
    """Creates a list parallel to class labels provided by fit()
    Args:
        X_train(list of list of obj): The list of training instances (samples)
        clases(list of obj): (Non-repeating) class labels
        key_value(obj): Given label in row
        total_classes(list of obj): Contains total number that classes appear in y_test
        inx(obj): current index of header
    Returns:
        values_list(list of obj): list with values parallel to class labels
    Notes:
        This function only works MyNaiveBayesClassifier.fit()
    """
    values_list = []
    for i in range(len(classes)):
        values_list.append(0)
    for i in range(len(X_train)):
        row = X_train[i]
        for index, value in enumerate(classes):
            if value in row and key_value == row[inx]:
                values_list[index] += 1
    for index, value in enumerate(total_classes):
        values_list[index] /= value
    return values_list

def unique(list1):
    """Finds unique instances in a list
    Args:
        list1(list of obj): list being examined for repeat instances
    Returns:
        unique_list(list of obj): list with unique instances
    Notes:
        This function can be used for any file
    """
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def get_column_no_header(table, table_index):
    """Extracts a column from a 2D list without needing a header
    Args:
        table(list of list of obj): 2D list in where column is going to be extracted from
        table index(obj): index where column should be extracted
    Returns:
        col(list of obj): extracted column
    Notes:
        This function can be used for any file
    """
    col_index = table_index
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col

def shuffle_parallel(array1, array2, random_seed):
    """shuffles two lists that are parallel to each other
    Args:
        array1(list of obj): list being shuffled
        array2(list of obj): list being shuffled
        random_seed(obj): number for random_seed
    Returns:
        list1, list2(list of obj): lists shuffled in parallel order
    Notes:
         This function can be used for any file
    """
    array1 = np.array(array1)
    array2 = np.array(array2)
    random.seed(random_seed)
    randomize = np.arange(len(array2))
    random.shuffle(randomize)
    array1 = array1[randomize]
    array2 = array2[randomize]
    list1 = array1.tolist()
    list2 = array2.tolist()
    return list1, list2

def compute_holdout_partitions_float(table, test_size):
    """Comptutes partitions for float
    Args:
        table(list of list of obj): 2D list
        test_size(obj): number being subtracted to find index
    Returns:
    Notes:
         This function can be used for any file
    """
    # randomize the table
    randomized = table[:] # copy the table
    n = len(table)
    test_size = 1 - test_size
    split_index = int(test_size * n) # 2/3 of randomized table is train, 1/3 is test
    return randomized[0:split_index], randomized[split_index:]

def compute_holdout_partitions_int(table, test_size):
    """Comptutes partitions for integer
    Args:
        table(list of list of obj): 2D list
        test_size(obj): number being subtracted to find index
    Notes:
         This function can be used for any file
    """
    # randomize the table
    randomized = table[:] # copy the table# 2/3 of randomized table is train, 1/3 is test
    split_index = len(randomized) - test_size
    return  randomized[0: len(randomized) - test_size], randomized[split_index:]
    
def randomize_in_place(a_list: list, parallel_list:list=None, random_state:int=None):
    """Shuffles the given list. If a parallel list is given, both lists are shuffled in parallel.

    Args:
        a_list (list of obj): list to be shuffled
        parallel_list (list of obj): OPTIONAL list to be shuffled in parallel with a_list
        random_state (int): integer used for seeding a random number generator for reproducible results
    """
    if random_state is not None:
        np.random.seed(random_state)
    for index in range(0, len(a_list)):
        random_index = np.random.randint(0, len(a_list))
        a_list[index], a_list[random_index] = a_list[random_index], a_list[index]
        if parallel_list != None:
            parallel_list[index], parallel_list[random_index] = parallel_list[random_index], parallel_list[index]

# MyDecisionTreeClassifier util functions
def compute_entropy(instances):
    value_counts = {}
    for instance in instances:
        instance_class = instance[len(instance)-1]
        if value_counts.get(instance_class) is None:
            value_counts[instance_class] = 1
        else:
            value_counts[instance_class] += 1
    value_priors = dict(sorted(value_counts.items(), key=lambda item: item[0]))
    num_instances = len(instances)
    for prior in list(value_priors.keys()):
        value_priors[prior] = value_priors[prior] / num_instances
    entropy = 0
    for prior in list(value_priors.values()):
        entropy += prior * math.log2(prior)
    entropy = entropy * -1
    return entropy

def select_attribute(current_instances, available_attributes, attribute_domain, header):
    E_new = {}
    E_start = compute_entropy(current_instances)
    num_instances = len(current_instances)
    for attribute in available_attributes:
        E_new[attribute] = 0
        attribute_partitions = partition_instances(current_instances, attribute, attribute_domain, header)
        for _, partition in attribute_partitions.items():
            attribute_value_count = len(partition)
            E_value = compute_entropy(partition)
            E_new[attribute] += (attribute_value_count/num_instances) * E_value
    attribute_gain = dict()
    for attribute in available_attributes:
        attribute_gain[attribute] = E_start - E_new[attribute]
    max_gain_value = max(list(attribute_gain.values()))
    selected_attribute = None
    for attribute_name, gain_value in attribute_gain.items():
        if gain_value == max_gain_value:
            selected_attribute = attribute_name
            break
    return selected_attribute

def partition_instances(instances, split_attribute, attribute_domains, header):
    """ partitions the passed in instances by the split instance
    Args:
        instances: (list of lists) table of data
        split_attribute: (string) attribute to split on
        attribute_domains: (dict) domains for the attributes
        header: (list) header of attributes
    Returns:
        partitions: (dict) new partitions by split attribute
    """
    partitions = {}
    att_index = header.index(split_attribute)
    att_domain = attribute_domains[split_attribute] # changed to split_attribute 
    partitions = {} 
    for attribute_value in att_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[att_index] == attribute_value:
                partitions[attribute_value].append(instance)
    return partitions 

def get_column_no_header(table, table_index):
    """Extracts a column from a 2D list without needing a header
    Args:
        table(list of list of obj): 2D list in where column is going to be extracted from
        table index(obj): index where column should be extracted
    Returns:
        col(list of obj): extracted column
    Notes:
        This function can be used for any file
    """
    col_index = table_index
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col

def get_frequencies_mod(table, index):
    """Gets frequency from a 2D list without needing a header
    Args:
        table(list of list of obj): 2D list in where column is going to be extracted from
        table index(obj): index where column should be extracted
    Returns:
        values(list of obj): list of values
        counts(list of obj): list of frequency of values
    Notes:
        This function can be used for any file, functions similar to get_frequencies()
    """
    col = get_column_no_header(table, index)
    col.sort() 
    values = []
    counts = []
    for value in col:
        if value in values:
            counts[-1] += 1 
        else: 
            values.append(value)
            counts.append(1)
    return values, counts 

def get_frequencies_given(col):
    col.sort() 
    values = []
    counts = []

    for value in col:
        if value not in values:
            values.append(value)
            counts.append(1)
        else:
            counts[-1] += 1 
    return values, counts 

def all_same_class(partition):
    values, _ = get_frequencies_mod(partition, -1)
    if len(values) == 1:
        return True
    else:
        return False

def majority_leaf(partition):
    """ calculates the majority class leaf node of the partition and returns the classification
    Args:
        partition: (list of lists) partitions of data
    Returns:
        attribute_class: (String) majority vote classification of partitions
    """
    col = []
    for item in partition:
        col.append(item[-1])
    values, counts = get_frequencies_given(col)
    max_num = max(counts)
    max_index = counts.index(max_num)
    attribute_class = values[max_index]

    return attribute_class

def tdidt(current_instances, available_attributes, attribute_domains, header):
    """ Recursive helper function to help form the tree
    Args:
        current_instances: (list of lists) table of data
        available_attributes: (list) available attributes for splitting
        attribute_domains: (dict) domains for the attributes
        header: (list) header of attributes
    Returns:
        tree: (nested list) decision tree created as a nested list
    """
    # select an attribute to split on
    attribute = select_attribute(current_instances, available_attributes, attribute_domains, header)
    available_attributes.remove(attribute) 
    # can't split again on this attribute
    tree = ["Attribute", attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, attribute, attribute_domains, header)
    # for each partition, repeat unless one of the following occurs (base case)
    case_three = False
    for att_value, att_partition in partitions.items():
        value_subtree = ["Value", att_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            attribute_class = att_partition[0][len(att_partition[0])-1]
            leaf_node = ["Leaf", attribute_class, len(att_partition), len(current_instances)]
            value_subtree.append(leaf_node)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            attribute_class = majority_leaf(att_partition)
            leaf_node = ["Leaf", attribute_class, len(att_partition), len(current_instances)]
            value_subtree.append(leaf_node)
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            values = []
            for att_value, att_partition in partitions.items():
                for item in att_partition:
                    if len(item) != 0:
                        values.append(item)
            attribute_class = majority_leaf(values)
            tree = ["Leaf", "case 3", attribute_class, len(att_partition), len(current_instances)]
            case_three = True
        else: # all base cases are false, recurse!!
            subtree = tdidt(att_partition, available_attributes.copy(), attribute_domains, header)
            value_subtree.append(subtree)
        #if case 3 == false, then append to tree
        if (case_three == False):
            tree.append(value_subtree)
    return tree

def predict_tree(X_test, tree):
    """
    Helper function for myclassifiers.predict()

    Args:
        X_test(list of list of obj): The list of testing samples
        tree(nested lists of obj): tree generated by myclassifiers.fit()
    Returns:
        leaf_node(list of obj): leaf node used to predict class label
    """
    if (tree[0] == "Attribute"):
        curr_string = tree[1]
        curr_index = int(curr_string[3])

        curr_value = X_test[curr_index]
        for i in range(2, len(tree)):
            if curr_value == tree[i][1]:
                tree = tree[i]
                return predict_tree(X_test, tree)
    elif ("Leaf" in tree):
        return tree
    tree = tree[2]
    return predict_tree(X_test, tree)

def common_instance(list):
    """finds the common instance in a list
    Args:
        list(list of obj): list to be looked at
    Returns:
        max()
    Notes:
         This function can be used for any file
    """
    return max(list, key = list.count)

