import myutils
import myutilsL
import math
import copy
import random

class MyRandomForestClassifier:

    def __init__(self) -> None:
        pass
    def fit():
        pass
    def predict():
        pass

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """

        # TODO: programmatically create a header (e.g. ["att0", "att1",
        # ...] and create an attribute domains dictionary)
        header = []
        x = 0
        for i in range(len(X_train[0])):
            header.append(f"att{x}")
            x += 1
        #att_domains = myutils.create_domains(header, X_train)
        att_domains = {}
        count = 0
        for item in header:
            curr_col = myutilsL.get_column(X_train, count)
            values, counts = myutilsL.get_frequencies(curr_col)
            att_domains[item] = values
            count+=1

        # next, I advise stitching X_train and y_train together
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # now, making a copy of the header because tdidt()
        # is going to modify the list
        available_attributes = header.copy()
        # recall: python is pass by object reference
        tree = myutilsL.tdidt(train, available_attributes, header, att_domains)
        #print("tree:", tree)
        self.tree = tree
        # note: the unit test will assert tree == interview_tree_solution
        # (mind the attribute value order)

        #pass # TODO: fix this

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for item in X_test:
            y_predict = myutilsL.predict_helper(item,self.tree)
            y_predicted.append(y_predict[1])

        return y_predicted
        #return [] # TODO: fix this

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None
        self.labels = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.labels, self.priors, self.posteriors = myutils.get_priors_and_posteriors(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        for i in range(len(X_test)):
            labels_probs = [] # parallel to self.labels
            priors_list = [] # parallel to self.labels
            for value in self.priors.values():
                priors_list.append(value)

            priors_list_idx = 0
            for key in self.posteriors.keys():
                probs_to_mul = []
                col_idx = 0
                for key2 in self.posteriors[key].keys():
                    for key3 in self.posteriors[key][key2].keys():
                        if X_test[i][col_idx] == key3:
                            probs_to_mul.append(self.posteriors[key][key2][key3])
                    col_idx += 1

                probs_to_mul.append(priors_list[priors_list_idx])
                probability = 1
                for j in range(len(probs_to_mul)):
                    probability = probability * probs_to_mul[j]

                labels_probs.append(probability)
                priors_list_idx += 1

            max = -1
            max_idx = 0
            for k in range(len(labels_probs)):
                if labels_probs[k] > max:
                    max = labels_probs[k]
                    max_idx = k
            y_predicted.append(self.labels[max_idx])

        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        # find what the most frequent instance in y-predicted should be
        # and store in self.most_common_label
        # self.most_common_label = myutils.common_instance(y_train)
        self.most_common_label = max(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for _ , _ in enumerate(X_test):
            y_predicted.append(self.most_common_label) 
        return y_predicted
        
class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        k_distances = []
        k_index = []

        # calculate distance
        for index, _ in enumerate(X_test):
            neighbor_index = []
            distances = []
            for i, train_instance in enumerate(self.X_train):
                distance = myutils.compute_euclidean_distance(train_instance, X_test[index])
                neighbor_index.append(i)
                distances.append(distance)
            k_distances.append(distances)
            k_index.append(neighbor_index)
        for index, _ in enumerate(k_distances):
            k_distances[index], k_index[index] = myutils.sort_lists(k_distances[index], k_index[index])

        for index, _ in enumerate(k_distances):
            k_distances[index] = k_distances[index][:self.n_neighbors]
            k_index[index] = k_index[index][:self.n_neighbors]
        return k_distances, k_index

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        _ , neighbor_index = MyKNeighborsClassifier.kneighbors(self, X_test) 
        y_predicted = []

        for index , _ in enumerate(neighbor_index):
            y_predicted_list = []
            temp_list = neighbor_index[index]
            for _ , instances in enumerate(temp_list):
                y_predicted_list.append(self.y_train[instances])
            y_predicted.append([myutils.common_instance(y_predicted_list)])  

        return y_predicted 


class MyRandomForestClassifier:
    """Represents a decision tree classifier.
    Attributes:
        N(int): Number of Classifiers to develop
        M(int): Number of better Classifiiers to use
        F(int): Size of random attribute subset
        X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
        trees(list of nested lists): list of trees generated
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, n = 20, m = 7, f = 2, seed = None):
        """Initializer for MyDecisionTreeClassifier.
        """

        self.N = n
        self.M = m
        self.F = f
        self.X_train = None 
        self.y_train = None
        self.trees = []
        self.seed = seed

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
         # fit() accepts X_train and y_train
        # TODO: calculate the attribute domains dictionary

        if (self.seed != None):
            random.seed(self.seed)
            
        n_trees = []
        accuracies = []
        for i in range(self.N):
            header = []
            attribute_domains = {}
            
            #loops through X_train and creates header
            for i in range(len(X_train[0])) :
                header.append("att" + str(i))
            

            #loops though header to form attribute domains dictionairy
            count = 0
            for item in header:
                curr_col = myutils.get_column2(X_train, count)
                values, counts = myutils.get_frequencies2(curr_col)
                attribute_domains[item] = values
                count+=1
                

            #stitching together X_train and y_train and getting available attributes
            train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
            available_attributes = header.copy()

            boot_train = myutils.compute_bootstrapped_sample(train)

            validation_set = []
            for row in train:
                if row not in boot_train:
                    validation_set.append(row)



            #forming tree
            tree = myutils.tdidt_forest(boot_train, available_attributes, attribute_domains, header, self.F)
            #print(tree)

            tree_dict = {}
            tree_dict["tree"] = tree
            y_test = []
            for row in validation_set:
                y_test.append(row.pop())
            
            y_predict = myutils.predict_tree(validation_set, tree)

            acc = myutils.get_accuracy(y_predict, y_test)
            tree_dict["acc"] = acc
            n_trees.append(tree_dict)
        

        sorted_trees = sorted(n_trees, key=lambda k: k['acc'], reverse=True)
        for i in range(self.M):
            self.trees.append(sorted_trees[i]["tree"])
        

        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for item in X_test:
            tree_predicts = []
            for tree in self.trees:
                y_predict = myutils.predict_helper(item,tree)
                tree_predicts.append(y_predict[1])
            
            y_predicted.append(max(set(tree_predicts), key=tree_predicts.count))

        return y_predicted