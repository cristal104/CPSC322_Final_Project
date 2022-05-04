import math

def get_column(table, col_idx):
    """ 
    Gets all the values of a certain column from a 2D list

    Args:
        table: 2D list of data
        col_idx: index of column
    
    Returns: a 1D list with all the values from the column
    """
    col = []
    for row in table: 
        if row[col_idx] != "NA":
            col.append(row[col_idx])
    return col

def get_frequencies(col):
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

    return values, counts # we can return multiple values in python
    # they are packaged into a tuple

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

def select_attribute(instances, attributes, header):
    """
    Selects an attribute to to split on by calculating entropy

    Args:
        instances: 2D list of data
        attributes: 1D list of available attributes
        header: 1D list of the column names

    Returns: the attribute to split on
    """
    num_instances = len(instances)
    e_new_list = []

    # groups data by each attribute
    for item in attributes: 
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
    attribute = attributes[min_index]

    return attribute

def partition_instances(header, instances, split_attribute, att_domains):
    """
    Creates partition for the instances

    Args:
        header: 1D list of attribute names
        instances: 2D list of instances
        split_attribute: attribute to split on
        att_domains: dictionary of all possible values of each attribute

    Returns: the parition
    """
    # this is a group by attribute domain
    # let's use a dictionary
    partitions = {} # key (attribute value): value (subtable)
    att_index = header.index(split_attribute) # e.g. level -> 0
    #print("DOMAINS:", att_domains)
    att_domain = att_domains[split_attribute] # e.g. ["Junior", "Mid", "Senior"]
    for att_value in att_domain:
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)
    return partitions

def find_majority_leaf(partition):
    """ 
    Finds the majority class leaf node 
    
    Args:
        partition: 2D list of partitions of data
  
    Returns: the classification
    """
    col = []
    for item in partition:
        col.append(item[-1])

    values, counts = get_frequencies(col)
    max_num = max(counts)
    max_index = counts.index(max_num)
    classification = values[max_index]

    return classification

def all_same_class(att_partition):
    """
    Checks to see if the instances in the parition all have the same class

    Args:
        att_partition: 2D list of instances

    Returns: True if all same class label, False otherwise
    """
    label = att_partition[0][-1]
    all_same = True
    for i in range(1, len(att_partition)):
        if att_partition[i][-1] == label:
            all_same = True
        else:
            all_same = False
            break
    return all_same

def tdidt(current_instances, available_attributes, header, att_domains):
    """
    Builds the decision tree using the tdit algorithm

    Args:
        current_instances: 2D list of instances
        available_attributes: 1D list of available attributes
        header: 1D list of attribute names
        att_domains: dictionary of all possible values of each attribute

    Returns: the tree
    """
    # basic approach (uses recursion!!):
    # print("available attributes:", available_attributes)

    # select an attribute to split on
    attribute = select_attribute(current_instances, available_attributes, header)
    # print("splitting on:", attribute)
    available_attributes.remove(attribute) # can't split on this again in
    # this subtree
    tree = ["Attribute", attribute] # start to build the tree!!

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(header, current_instances, attribute, att_domains)
    # print("partitions:", partitions)
    for attribute_value, attribute_partition in partitions.items():
        value_subtree = ["Value", attribute_value]

        #   CASE 1: all class labels of the partition are the same => make a leaf node
        if len(attribute_partition) > 0 and all_same_class(attribute_partition):
            # print("CASE 1 all same class")
            attribute_class = attribute_partition[0][-1]
            leaf_node = ["Leaf", attribute_class, len(attribute_partition), len(current_instances)]
            value_subtree.append(leaf_node)
            tree.append(value_subtree)
        
        #   CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(attribute_partition) > 0 and len(available_attributes) == 0:
            # print("CASE 2 no more attributes")
            classification = find_majority_leaf(attribute_partition)
            leaf_node = ["Leaf", classification]
            value_subtree.append(leaf_node)
            tree.append(value_subtree)
        
        #   CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(attribute_partition) == 0:
            # print("CASE 3 empty partition")
            # "backtrack" to replace the attribute node with a majority leaf node by replacing the current "Attribute" node w/
            # a majority vote leaf node
            values = []
            for attribute_value, partition in partitions.items():
                for item in partition:
                    if len(item) != 0:
                        values.append(item)
 
            classification = find_majority_leaf(values)
            tree = ["Leaf", classification]
        else: # none of the previous conditions were true... recurse!
            subtree = tdidt(attribute_partition, available_attributes.copy(), header, att_domains)
            # note the copy
            # append subtree to value_subtree and tree appropriately
            value_subtree.append(subtree)
            tree.append(value_subtree)
    return tree

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