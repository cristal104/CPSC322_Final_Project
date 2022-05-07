import csv
import math
import numpy as np
import random
import copy

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

def get_column2(table, col_index):
    """ gets the column from a passed in col_name
    Args:
        table: (list of lists) table of data to get column from
        col_index: index of column in table
    Returns:
        col: (list) column wanted
    """
    col = []
    for row in table: 
        # ignore missing values ("NA")
        if row[col_index] != "NA":
            col.append(row[col_index])
    return col

def change_isFraud(table, header, col_name):
    """Changes the binary classification for "isFraud"

Args:
    table(list of list of obj): 2D list for dataset
    header(list of obj): 1D list of class attributes parallel to table

Notes:
    This function only works to change the binary class labels for "isFraud" class attributes
"""
    col_index = header.index(col_name)
    for row in table:
        value = row[col_index]
        if value == '0':
            row[col_index] = "no"
        if value == '1':
            row[col_index] = "yes"

def dictionary_to_list(dictionary):
    '''fits a dictionary into two lists, one list for keys and one list of values 
    Args:
        dictionary(dict of obj): dictionary with key-value pairs
    Returns:
        x_list(list of obj): list with keys from dictionary
        y_list(list of obj): list with values from dictionary
    '''
    x_list = list(dictionary.keys())
    y_list = list(dictionary.values())
    return x_list, y_list

def get_frequencies(table, header, col_name):
    """Gets the frequency of instances and returns results in a list
    Args:
        table(list of list of obj): 2D list with dataset
        header(list of obj): 1D list parallel to table
        col_name(obj): Name of column being analyzed
    Returns:
        values(list of obj): list of unique instances in col_name
        counts(list of obj): lisf of count of values in col_name
    """
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
    """Deletes a given column from a 2D list
    Args:
        table(list of list of obj: 2D list holding dataset
        header(list of obj): 1D list parallel to table
        col_name(obj): name of column being deleted
    Notes: 
        This function does not return a new table and header
    """
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
        table_index(obj): index where column should be extracted
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
    """
    Calculates entropy

    Args:
        instances: 2D list of data

    Returns: entropy value
    """
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
    """
    Selects an attribute to split on by calculating entropy

    Args:
        current_instances: 2D list of data
        available_attributes: 1D list of available attributes
        attribute_domains: dictionary of all possible values of each attribute
        header: 1D list of the column names

    Returns: the attribute to split on
    """
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
    """Gets frequency from a 2D list given a single column
    Args:
        col(list of obj): single column
    Returns:
        values(list of obj): list of values
        counts(list of obj): list of frequency of values
    Notes:
        This function can be used for any file, functions similar to get_frequencies()
    """
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
    """
    Checks to see if the instances in the partition all have the same class

    Args:
        partition: 2D list of instances

    Returns: True if all same class label, False otherwise
    """
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
    """ Builds the decision tree using the tdit algorithm
    Args:
        current_instances: 2D list of instances
        available_attributes: 1D list of available attributes
        attribute_domains: dictionary of all possible values of each attribute
        header: 1D list of attribute names
    Returns: the tree
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

def compute_bootstrapped_sample(table, seed_num=None):
    n = len(table)
    sample = []
    for _ in range(n):
        rand_index = random.randrange(0, n)
        sample.append(table[rand_index])
    return sample

def compute_random_subset(values, num_values):
    shuffled = values[:]
    random.shuffle(shuffled)
    return shuffled[:num_values]

def calc_majority_leaf(partition):
    """ 
    Finds the majority class leaf node 
    
    Args:
        partition: 2D list of partitions of data
  
    Returns: the classification
    """
    col = []
    for item in partition:
        col.append(item[-1])

    values, counts = get_frequencies2(col)
    max_num = max(counts)
    max_index = counts.index(max_num)
    classification = values[max_index]

    return classification

def tdidt_forest(current_instances, available_attributes, attribute_domains, header, F):
    """ Recursive helper function to help form the tree
    Args:
        current_instances: (list of lists) table of data
        available_attributes: (list) available attributes for splitting
        attribute_domains: (dict) domains for the attributes
        header: (list) header of attributes
    Returns:
        tree: (nested list) decision tree created as a nested list
    """
    atts = compute_random_subset(available_attributes, F)
    
    # select an attribute to split on
    split_attribute = select_attribute2(current_instances, atts, header)

    # remove split attribute from available attributes
    # because, we can't split on the same attribute twice in a branch

    available_attributes.remove(split_attribute) # Python is pass by object reference!!
    atts.remove(split_attribute)
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, attribute_domains, header)

    # for each partition, repeat unless one of the following occurs (base case)
    Skip = False
    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            leaf_node = ["Leaf", partition[0][-1]]
            values_subtree.append(leaf_node)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            classification = calc_majority_leaf(partition)
            leaf_node = ["Leaf", classification]
            values_subtree.append(leaf_node)

        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:

            values = []
            #loops trhough each current partition and further each item in the partitions
            for attribute_value, partition in partitions.items():
                for item in partition:

                    #checks if the partition isn't empty and adds them to a list
                    if len(item) != 0:
                        values.append(item)

            #calculates the majority leaf node of the values 
            classification = calc_majority_leaf(values)

            #sets the current attribute to a leaf node
            tree = ["Leaf", classification]
            Skip = True
        else: # all base cases are false, recurse!!
            subtree = tdidt_forest(partition, available_attributes.copy(), attribute_domains, header,F)
            values_subtree.append(subtree)
        #if case 3 didn't occur, the tree appends the values subtree
        if (Skip == False):
            tree.append(values_subtree)
    return tree


def get_frequencies2(col):
    """ 
    Computes the frequencies of each value

    Args:
        col: (list) column name of frequencies to find

    Returns:
        values: (list) values in col_name
        counts: (list) paralel list to values list of frequency counts
    """

    col.sort() # inplace
    values = []
    counts = []

    for value in col:
        if value in values: # seen it before
            counts[-1] += 1 # ok because the list is sorted
        else: # haven't seen it before
            values.append(value)
            counts.append(1)

def group_by(table, col_index):
    """ 
    Creates subtables of table based on unique values

    Args:
        table: 2D list of data
        col_index: index of column in the table
    
    Returns:
        group_names: list of group label names
        group_subtables: 2D list of each group subtable
    """
    col = get_column(table, col_index)

    group_names = sorted(list(set(col))) # 75, 76, 77
    group_subtables = [[] for _ in group_names] # [[], [], []]

    for row in table:
        group_value = row[col_index]
        # which subtable does this row belong?
        group_index = group_names.index(group_value)
        group_subtables[group_index].append(row.copy()) # shallow copy

    return group_names, group_subtables


def select_attribute2(instances, available_attributes, header):
    """
    Selects an attribute to to split on by calculating entropy

    Args:
        instances: 2D list of data
        available_attributes: 1D list of available attributes
        header: 1D list of the column names

    Returns: the attribute to split on
    """
    num_instances = len(instances)
    e_new_list = []

    # groups data by each attribute
    for item in available_attributes: 
        group_names, group_subtables = group_by(instances, header.index(item))
        e_value_list = []
        num_values = []

        # loops through the group subtable and further groups by class name
        for j in range(len(group_subtables)):
            group = group_subtables[j]
            num_attributes = len(group)
            num_values.append(num_attributes)
            class_names, class_subtables = group_by(group, len(group[0])-1)
            e_value = 0

            if (len(class_subtables) == 1):
                    e_value = 0
            else :
                # calculates the entropy
                for k in range(len(class_subtables)):
                    class_num = len(class_subtables[k]) / num_attributes
                    e_value -= (class_num) * (math.log2(class_num))
            e_value_list.append(e_value)
        e_new = 0

        #calculates e_new
        for l in range (len(e_value_list)):
            e_new += e_value_list[l] * (num_values[l] / num_instances)
        e_new_list.append(e_new)

    # selects attribute with smallest entropy
    min_entropy = min(e_new_list)
    min_index = e_new_list.index(min_entropy)
    attribute = available_attributes[min_index]

    return attribute

def get_accuracy(y_predicted, y_test):
    """ 
    Calculates accuracy of classifier
    
    Args:
        y_predicted: list of predicted class labels
        y_test (parallel to y_predicted): list of actual class labels
    
    Returns: accuracy of classifier 
    """
    correct_count = 0
    for i in range(len(y_predicted)):
        if (y_predicted[i] == y_test[i]):
            correct_count += 1
    
    return correct_count / len(y_predicted)

def predict_helper(X_test, tree) :
    """ 
    Helper function for predict

    Args:
        X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        tree: current tree
    
    Returns: leaf node found for classification
    """

    if (tree[0] == "Attribute"):
        curr_string = tree[1]
        curr_idx = int(curr_string[3])
    
        curr_value = X_test[curr_idx]
        for i in range(2, len(tree)):
            if curr_value == tree[i][1]:
                curr_tree = tree[i]
                return predict_helper(X_test, curr_tree)
                
    # leaf node
    elif ("Leaf" in tree):
        return tree
    
    tree = tree[2]
    return predict_helper(X_test, tree)


def get_priors_and_posteriors(X_train, y_train):
    '''
    Creates labels(1D list of all possible class labels), priors(dict), and posteriors(dict)
    
    Args:
        X_train(list of list of obj): The list of training instances (samples)
            The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train)
            The shape of y_train is n_train_samples
    
    Returns: labels, priors, and posteriors
    '''
    labels = []
    for i in range(len(y_train)):
        if y_train[i] not in labels:
            labels.append(y_train[i])

    priors = {}
    posteriors = {}
    for i in range(len(labels)):
        priors[labels[i]] = 0
        posteriors[labels[i]] = {}

    for i in range(len(y_train)):
        for key in priors.keys():
            if y_train[i] == key:
                priors[key] += 1

    for key in priors.keys():
        priors[key] = priors[key] / len(y_train)

    attributes = {}
    i = 1
    for j in range(len(X_train[0])):
        attributes[f"att{i}"] = {}
        i += 1

    col_idx = 0
    for key in attributes.keys():
        for i in range(len(X_train)):
            attributes[key][X_train[i][col_idx]] = 0
        col_idx += 1

    for key in posteriors.keys():
        posteriors[key] = copy.deepcopy(attributes)

    # posteriors
    labels_instances_count = [] # parallel to labels
    for key in posteriors.keys():
        indices = []
        for i in range(len(y_train)):
            if y_train[i] == key:
                indices.append(i)
        labels_instances_count.append(len(indices))
        col_idx = 0
        for key2 in posteriors[key].keys():
            for key3 in posteriors[key][key2].keys():
                for i in range(len(indices)):
                    if X_train[indices[i]][col_idx] == key3 and y_train[indices[i]] == key:
                        posteriors[key][key2][key3] = posteriors[key][key2][key3] + 1
            col_idx += 1

    i = 0
    for key in posteriors.keys():
        for key2 in posteriors[key].keys():
            for key3 in posteriors[key][key2].keys():
                posteriors[key][key2][key3] = posteriors[key][key2][key3] / labels_instances_count[i]
        i += 1

    return labels, priors, posteriors