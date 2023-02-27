# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from graphviz import Source

# Members: Erik Hale(emh170004) and Shiva Kumar (sak220007)


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    uniqueValue = set()
    indexDictionary = {}
    for i in range(len(x)):
        # Unique value has not been found
        if x[i] not in uniqueValue:
            uniqueValue.add(x[i])
            indexDictionary[x[i]] = np.array([i])
        # Add the index to the unique value
        else:
            indexDictionary[x[i]] = np.append(indexDictionary[x[i]], i)

    return indexDictionary
    raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # REMEMBER: The more the classes are alike, the higher this is, but this is also in the
    # ... negatives, so we may need to negate this value
    H = 0

    # Calculate the total size of the y
    totalSize = 0
    for i in y:
        totalSize += i.size

    # Calculate entropy
    for uniqueValue in y:
        probability = uniqueValue.size/totalSize
        H += probability * np.log2(probability)

    # H = H * -1
    H = -H

    return H


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    # It also saying mutual_information(x, y) = entropy(y) - entropy(...) <- still have to look into this
    # I(x, y) = H(y) + H(x) - H(x, y)
    # INSERT YOUR CODE HERE

    # Find the unique elements for x

    partitionedElements = partition(x)

    # Entropy before splitting the set
    entropyBefore = entropy(partitionedElements)
    maxInformationGain = -np.inf
    attributeValue = np.inf

    # For every element in the array subtract it from the array.
    # Then compute the entropy and calculate information-gain
    for element in partitionedElements:
        # Subtract the current element from dataSet
        newX = np.array([])
        # For every value that is not the element add to newX
        for count in x:
            if count != element:
                newX = np.append(newX, count)

        # Calculate Entropy after
        newPartition = partition(newX)
        entropyAfter = entropy(newPartition)
        currentInfoGain = entropyBefore - entropyAfter
        # If the information gain is greather than the maxInformationGain
        if currentInfoGain > maxInformationGain:
            maxInformationGain = currentInfoGain
            attributeValue = element

    # Return the attribute with the highest information gain
    return attributeValue

    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    # For "0/1 not in y", do we need to return an array the size of y? <- Probably not because it will come down
    # ... to one value(?) I assume that they can all be classified under that variable
    ''' Might try to make this work for more than 0 or 1 as well for multiple classes'''

    # Will be used to store the information in nested dictionary
    attributeValue = 0
    attribute = 0
    # Termination cases
    if (0 not in y):

        return 1
    elif (1 not in y):
        return 0
    elif ((len(attribute_value_pairs) == 1) or (max_depth == depth)):
        return 0 if np.count_nonzero(y == 0) > np.count_nonzero(y == 1) else 1
    else:
        maxEntropy = 0
        for i in range(len(attribute_value_pairs)):

            currentAttribute = attribute_value_pairs[i][0]

            partitionElements = partition(x[:, currentAttribute])
            # Calculate entropy on current attribute
            currentEntropy = entropy(partitionElements)
            if currentEntropy > maxEntropy:
                # The attribute has a greater entropy
                maxEntropy = currentEntropy
                attribute = currentAttribute
                # Gets the best attribute based on information-gain
                attributeValue = mutual_information(x[:, currentAttribute], y)

    # Delete attribute-value pair from binary tree
    pair = [attribute, attributeValue]
    newAttribute_value_pairs = []
    for valuePair in attribute_value_pairs:
        if valuePair[0] != pair[0] and valuePair[1] != pair[1]:
            newAttribute_value_pairs.append(valuePair)

    # So now we have a new binary tree without the best attribute-value pair
    newAttribute_value_pairs = np.array(newAttribute_value_pairs)

    # Now we will split the x into two arrays.
    # One array will have the x with the attribute that we test and one without the attribute
    datawithattribute = []
    ywithattribute = []
    datawithoutattribute = []
    ywithoutattribute = []
    for index, value in enumerate(x):
        if ((value[attribute]) == attributeValue):
            datawithattribute.append(value)
            ywithattribute.append(y[index])

        else:
            datawithoutattribute.append(value)
            ywithoutattribute.append(y[index])

    ywithattribute = np.array(ywithattribute)
    ywithoutattribute = np.array(ywithoutattribute)
    datawithattribute = np.array(datawithattribute)
    datawithoutattribute = np.array(datawithoutattribute)
    nestedDictionary = {(attribute, attributeValue, False): {
    }, (attribute, attributeValue, True): {}}

    # Split x into two parts: one with attribue value and one without it
    nestedDictionary[(attribute, attributeValue, False)] = id3(
        datawithoutattribute, ywithoutattribute, attribute_value_pairs, depth + 1, max_depth)
    nestedDictionary[(attribute, attributeValue, True)] = id3(
        datawithattribute, ywithattribute, attribute_value_pairs, depth + 1, max_depth)
    # print(nestedDictionary)
    return nestedDictionary

    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    if tree == 0:
        return 0
    elif tree == 1:
        return 1
    else:

        # Iterates the nested dictionary
        for key, value in tree.items():
            attribute = key[0]
            valuetoCheck = key[1]

            # If x[attribute] then recursively call the predict_example
            if (key[2] == (x[attribute] == valuetoCheck)):
                return predict_example(x, value)

    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    return (1 / len(y_true)) * sum(y_true != y_pred)


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print(
            '+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def confusionMatrix(y_true, y_pred):
    '''
    Prints the Confusion matrix (TPR, FPR, TNR, FNR)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    TNR = TN / (TN + FP)
    This function assumes that the input is binary
    '''
    total_cases = len(y_true)
    # Get the TP, TN, FN, FP, and print confusion matrix
    TP, TN, FN, FP = 0, 0, 0, 0
    for i in range(total_cases):
        if (y_true[i] == 1 and y_pred[i] == 1):
            TP += 1
        elif (y_true[i] == 1 and y_pred[i] == 0):
            FN += 1
        elif (y_true[i] == 0 and y_pred[i] == 0):
            TN += 1
        elif (y_true[i] == 0 and y_pred[i] == 1):
            FP += 1
    print("TP: ", TP, "| FN: ", FN)
    print("--------------------------------------------")
    print("FP: ", FP, "| TN: ", TN)
    print("\n")
    TPR = (TP)/(TP + FN)
    TNR = (TN)/(TN + FP)
    FPR = (FP)/(FP+TN)
    FNR = (FN)/(FN + TP)

    print("TPR: ", TPR, "% | FNR: ", FNR, "%")
    print("----------------------")
    print("FPR: ", FPR, "% | TNR: ", TNR, "%")
    print("\n")
    # Get the rates if needed:
    '''
    print ("TPR: ", TP / (TP + FN), "%")
    print ("TNR: ", TN / (TN + FP), "%")
    print ("FPR: ", FP / (FP + TN), "%")
    print ("FNR: ", FN / (FN + TP), "%")
    '''


if __name__ == '__main__':
    # Part a: Test on all 3 monk training sets
    for dataset in range(1, 4):
        # Create the file names for the set of data
        fileNameTrain = './data/monks-' + str(dataset) + '.train'
        fileNameTest = './data/monks-' + str(dataset) + '.test'
        # Load the training data
        M = np.genfromtxt(fileNameTrain, missing_values=0,
                          skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]

        # Load the test data
        M = np.genfromtxt(fileNameTest, missing_values=0,
                          skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]

        # Calculating inital attribute_value_pair binary tree
        attribute_value = []
        # i represents the attribute
        for i in range(len(Xtrn[0])):
            # Partitions each Element
            partitionedElements = partition(Xtrn[:, i])
            for element in partitionedElements:
                attribute_value.append([i, element])
        attribute_value = np.array(attribute_value)

        # initialize the arrays to hold the training and testing data for Monk Dataset
        saveTrainError = np.zeros(10)
        saveTestError = np.zeros(10)
        counter = np.zeros(10, dtype=int)

        # Find decision tree for depths 1 through 10
        for maxdepth in range(1, 11):
            print("Depth is ", maxdepth)
            decision_tree = id3(
                Xtrn, ytrn, attribute_value, max_depth=maxdepth)
            visualize(decision_tree)

            # Compute the test error
            y_pred = [predict_example(x, decision_tree) for x in Xtst]
            tst_err = compute_error(ytst, y_pred)

            # Compute the train error
            trainypred = [predict_example(x, decision_tree) for x in Xtrn]
            train_err = compute_error(ytrn, trainypred)
            print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
            print('Train Error = {0:4.2f}%.'.format(train_err * 100))

            # append data
            saveTrainError[maxdepth - 1] = train_err
            saveTestError[maxdepth - 1] = tst_err
            counter[maxdepth - 1] = maxdepth

        # Graph the plots in MatPlotLib
        title = "Graph for Monk-" + str(dataset)
        plt.title(title)
        plt.plot(counter, saveTrainError, color="red", label='Training Error')
        plt.plot(counter, saveTestError, color="blue", label='Testing Error')
        plt.legend()
        plt.show()

        # monk 1 data set

        if (dataset == 1):
            # Part B: Report Confusion Matrix
            print("Monk 1 Dataset")
            fileNameTrain = './data/monks-1.train'
            fileNameTest = './data/monks-1.test'
            # Load the training data
            M = np.genfromtxt(fileNameTrain, missing_values=0,
                              skip_header=0, delimiter=',', dtype=int)
            ytrn = M[:, 0]
            Xtrn = M[:, 1:]

            # Load the test data
            M = np.genfromtxt(fileNameTest, missing_values=0,
                              skip_header=0, delimiter=',', dtype=int)
            ytst = M[:, 0]
            Xtst = M[:, 1:]

            # Calculating inital attribute_value_pair binary tree
            attribute_value = []
            # i represents the attribute
            for i in range(len(Xtrn[0])):
                # Partitions each Element
                partitionedElements = partition(Xtrn[:, i])
                for element in partitionedElements:
                    attribute_value.append([i, element])
            attribute_value = np.array(attribute_value)
            # For depths 1 and 2
            for i in range(1, 3):
                decision_tree = id3(
                    Xtrn, ytrn, attribute_value, max_depth=i)
                visualize(decision_tree)
                y_pred = [predict_example(x, decision_tree) for x in Xtst]
                tst_err = compute_error(ytst, y_pred)
                confusionMatrix(ytst, y_pred)

        # Part B for SPECT dataset
        elif (dataset == 2):
            print("SPEC Data Set")
            fileNameTrain = './data/SPECT.train'
            fileNameTest = './data/SPECT.test'
            # Load the training data
            M = np.genfromtxt(fileNameTrain, missing_values=0,
                              skip_header=0, delimiter=',', dtype=int)
            ytrn = M[:, 0]
            Xtrn = M[:, 1:]

            # Load the test data
            M = np.genfromtxt(fileNameTest, missing_values=0,
                              skip_header=0, delimiter=',', dtype=int)
            ytst = M[:, 0]
            Xtst = M[:, 1:]

            # Calculating inital attribute_value_pair binary tree
            attribute_value = []
            # i represents the attribute
            for i in range(len(Xtrn[0])):
                # Partitions each Element
                partitionedElements = partition(Xtrn[:, i])
                for element in partitionedElements:
                    attribute_value.append([i, element])
            attribute_value = np.array(attribute_value)
            # For depths 1 and 2
            for i in range(1, 3):
                decision_tree = id3(
                    Xtrn, ytrn, attribute_value, max_depth=i)
                visualize(decision_tree)
                y_pred = [predict_example(x, decision_tree) for x in Xtst]
                tst_err = compute_error(ytst, y_pred)
                confusionMatrix(ytst, y_pred)

        # Part B for Blood Transfusion Dataset
        # The feature is the last column
        else:
            print("Blood-Transfusion Dataset")
            M = np.genfromtxt("./data/transfusion.data", missing_values=0,
                              skip_header=0, delimiter=',', dtype=int)
            y = M[:, -1]
            x = M[:, :-1]

            Xtrn, Xtst, ytrn, ytst = train_test_split(
                x, y, random_state=100, test_size=0.4, shuffle=True)

            # Calculating inital attribute_value_pair binary tree
            attribute_value = []
            # i represents the attribute
            for i in range(len(Xtrn[0])):
                # Partitions each Element
                partitionedElements = partition(Xtrn[:, i])
                for element in partitionedElements:
                    attribute_value.append([i, element])
            attribute_value = np.array(attribute_value)
            # For depths 1 and 2
            for i in range(1, 3):
                decision_tree = id3(
                    Xtrn, ytrn, attribute_value, max_depth=i)
                visualize(decision_tree)
                y_pred = [predict_example(x, decision_tree) for x in Xtst]
                tst_err = compute_error(ytst, y_pred)
                confusionMatrix(ytst, y_pred)

            # Part (c): Using default SciKit Learn
        dtskl = DecisionTreeClassifier()

        # Train the classifier
        dtskl.fit(Xtrn, ytrn)

        # display the classifier
        # Turns the SciKit into a dot file.
        dataName = ''
        if (dataset == 1):
            dataName = 'Monks-1'
        elif (dataset == 2):
            dataName = 'SPECT'
        else:
            dataName = 'Transfusion'
        dot_data = tree.export_graphviz(dtskl, out_file=None, filled=True)
        graph = Source(dot_data, format="png")
        graph

        graph.render("decision_tree_graphviz_" +
                     dataName, "graphviz_dataset_" + dataName)

        # predict the testing set
        yPreddtskl = dtskl.predict(Xtst)

        # Compute the error
        skl_tst_err = compute_error(ytst, yPreddtskl)
        print("Test Error for SciKit Learn: {0:4.2f}%.".format(
            skl_tst_err * 100))

        # Find Confusion Matrix
        TN, FP, FN, TP = confusion_matrix(ytst, yPreddtskl).ravel()
        print("TP: ", TP, " | FN: ", FN)
        print("------------------")
        print("FP: ", FP, " | TN: ", TN)
        print("\n")
        TPR = (TP)/(TP + FN)
        TNR = (TN)/(TN + FP)
        FPR = (FP)/(FP+TN)
        FNR = (FN)/(FN + TP)

        print("TPR: ", TPR, "% | FNR: ", FNR, "%")
        print("----------------------")
        print("FPR: ", FPR, "% | TNR: ", TNR, "%")
        print("\n")
