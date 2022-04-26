import myutils
import math
import copy

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
        # create attribute domain dictionary and header beforehand
        header = []
        attribute_domain = {}
        for i in range(len(X_train[0])):
            header.append(f"att{i}")

        for index, value in enumerate(header):
            col = myutils.get_column_no_header(X_train, index)
            values, _ = myutils.get_frequencies_given(col)
            attribute_domain[value] = values
        
        # create tree :)
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = header.copy()
        tree = myutils.tdidt(train, available_attributes, attribute_domain, header)
        self.tree = tree
            
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for value in X_test:
            y_predicted_values = myutils.predict_tree(value, self.tree)
            y_predicted.append(y_predicted_values[1])
        return y_predicted

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