import myutils
import math
import copy

class MyRandomForestClassifier():

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
        priors(dictionary): The prior probabilities computed for each
            label in the training set.
        posteriors(dictionary of dictionary): The posterior probabilities computed for each
            attribute value/label pair in the training set.
        header(list of obj): The labels given to the columns in X_train
        classes(list of obj): The class labels given from y_train

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, header=None):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None
        self.classes = None
        if header is None:
            header= []
        self.header = copy.deepcopy(header)

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
        # 1. Calculate PRIORS and store in dictionary
        unique_classes = myutils.unique(y_train)
        prior_classes = {}
        for _ , value in enumerate(unique_classes):
            prior_classes[value] = 0
        for _, value in enumerate(y_train):
            prior_classes[value] += 1
        copy_prior_classes = copy.deepcopy(prior_classes)
        _ , total_classes = myutils.dictionary_to_list(copy_prior_classes)
        for key in prior_classes:
            prior_classes[key] /= len(y_train)
        self.priors = prior_classes

        # 2. Calculate POSTERIORS and store in dictionary
        classes, _ = myutils.dictionary_to_list(self.priors)
        self.classes = classes
        keys_list = []
        sub_dict = {}
        main_dict= {}
        for index, value in enumerate(self.header):
            column = myutils.get_column_no_header(X_train, index)
            keys_list = myutils.unique(column)
            for _, val in enumerate(keys_list):
                value_classes_list = myutils.create_list(X_train, self.classes, val, total_classes, index)
                sub_dict[val] = value_classes_list
            main_dict[value] = sub_dict
            sub_dict = {}

        compare_list = myutils.get_column(X_train, self.header, self.header[-1])
        if compare_list == y_train:
            del main_dict[self.header[-1]]
            del self.header[-1]

        self.posteriors = main_dict

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        class_calculation = []
        sublist = []
        calculations_dict = {}
        for X_index, _ in enumerate(X_test):
            current_X_test_row = X_test[X_index]
            for i in range(len(self.classes)):
                for index, value in enumerate(current_X_test_row):
                    sublist = self.posteriors[self.header[index]][value]
                    class_calculation.append(sublist[i])
                class_calculation.append(self.priors.get(self.classes[i]))
                calculations_dict[self.classes[i]] = math.prod((class_calculation))
                class_calculation = []
            max_key = max(calculations_dict, key=calculations_dict.get)
            y_predicted.append(max_key)
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
        self.most_common_label = myutils.common_instance(y_train)
        print(self.most_common_label)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for i, _ in enumerate(X_test):
            y_predicted.append([self.most_common_label]) 
        return y_predicted