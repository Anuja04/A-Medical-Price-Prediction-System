'''
Implementation of Random Forest Regression to predict the medical prices.
'''
# @author: Anuja Tike, SJSU, May 2018

import numpy as np
import pandas as pd
from random import randrange

# Column labels.
# These are used only to print the tree.
header = ["DRG", "Region","TotalDischarge","Houseprice","label"]


#__________________________________Decision Tree_______________________________________________________

#Function to get prediction at leaf node
def getLeaf(prediction):
    return prediction


# Function to check conditions in a regression tree whether to a record reached at leaf node
# or if not then again split the records in true and false records based on a condition.
def regression(row, node):

    # Base case: when reached at leaf node
    if isinstance(node, Leaf):
        return node.predictions

    # Decision to be done whether to go to the true-branch or the false-branch.
    # Compare the feature / value stored in the node for the record under consideration.
    if node.question.match(row):
        return regression(row, node.trueBranch)
    else:
        return regression(row, node.falseBranch)

def printTree(node, spacing=""):
    """tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    printTree(node.trueBranch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    printTree(node.falseBranch, spacing + "  ")

def buildRegressionTree(records,maxDepth,minSamples,depth):
    # Builds the tree.
    # Tries partitioing the dataset on each of the unique attribute,
    # calculate the information gain, and return the question that produces the highest gain.
    gain, questionToAsk = findBestSplit(records)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.

    if gain == 0:
        return Leaf(records)

    if depth>=maxDepth:
        return  Leaf(records)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    trueRecords, falseRecords = partition(records, questionToAsk)

    if len(trueRecords)<=minSamples:
        return Leaf(records)

    else:
        # Recursively build the true branch.
        trueBranch = buildRegressionTree(trueRecords, maxDepth, minSamples,depth+1)

    if len(falseRecords)<=minSamples:
        return  Leaf(records)

    else:
        # Recursively build the false branch.
        falseBranch = buildRegressionTree(falseRecords, maxDepth, minSamples,depth+1)



    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return DecisionNode(questionToAsk, trueBranch, falseBranch)

class DecisionNode:

    """A Decision Node asks a question. This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 trueBranch,
                 falseBranch):
        self.question = question
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch

class Leaf:
    """A Leaf node predicts data.avg of the labels in rows from the training data that reach this leaf.
    """

    def __init__(self, records):
        self.predictions = avg_counts(records)



def findBestSplit(records):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""

    bestGain = 0  # keep track of the best information gain
    bestQuestion = None  # keep train of the feature / value that produced it
    currentVariance = varianceCal(records)
    nFeatures = len(records[0]) - 1  # number of columns

    for col in range(nFeatures):  # for each feature

        values = set([row[col] for row in records])  # unique values in the column

        for val in values:  # for each value

            questionToAsk = QuestionClass(col, val)

            # try splitting the dataset
            trueRecords, falseRecords = partition(records, questionToAsk)

            # Skip this split if it doesn't divide the dataset.
            if len(trueRecords) == 0 or len(falseRecords) == 0:
                continue

            # Calculate the information gain from this split
            gain = infoGain(trueRecords, falseRecords, currentVariance)


            if gain >= bestGain:
                bestGain, bestQuestion = gain, questionToAsk

    return bestGain, bestQuestion



def infoGain(left, right, currentVariance):

    """Information Gain: The variance of the starting node, minus the weighted impurity of two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    info_Gain=currentVariance - p * varianceCal(left) - (1 - p) * varianceCal(right)

    return int(round(info_Gain))

def varianceCal(records):

    yList=[]

    for i in range(len(records)):
        yList.append(records[i][4])

    #If no rows are matching the condition
    if len(yList)==0:
        return 0

    else:

        yMean= np.mean(yList)

        variance=0
        for i in range(len(yList)):
            variance=variance + (yList[i]-yMean)**2


        variance=variance/len(yList)


    return int(round(variance))

def partition(records, question):

    """Partitions a dataset. For each row in the dataset, check if it matches the question or not. If so, add it to 'trueRows', otherwise, add it to 'falseRows'.
    """
    trueRows, falseRows = [], []
    for row in records:
        if question.match(row):
            trueRows.append(row)
        else:
            falseRows.append(row)
    return trueRows, falseRows

class QuestionClass:

    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for drg) and a
    'column value' (e.g., 65). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):

        # Compare the feature value in an example to the feature value in this question.
        val = example[self.column]
        if isNumeric(val):
            return val <= self.value
        else:
            return val == self.value

    def __repr__(self):

        # A helper method to print the question in a readable format.
        condition = "=="
        if isNumeric(self.value):
            condition = "<="

        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

def isNumeric(value):
    """Test if a value is numeric."""

    return isinstance(value, int) or isinstance(value, float)

def avg_counts(rows):

    """Calculates the avg of all the labels of rows which reached to leaf."""

    predictedValue=0
    for i in range(len(rows)):
        predictedValue=predictedValue+rows[i][4]

    predictedValue=predictedValue/len(rows)

    return int(round(predictedValue))

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    print("\nPrinting unique vals in a column:\n")
    print(set([row[col] for row in rows]))
    return set([row[col] for row in rows])



#________________________________________Random Forest______________________________________________

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

def randomForest(trainData, n_trees, sample_size,max_depth, min_size, depth):
    trees=list()
    for i in range(n_trees):
        sample = subsample(trainData, sample_size)
        tree = buildRegressionTree(sample,max_depth, min_size, depth)
        printTree(tree)
        trees.append(tree)

    return trees

# Make a prediction with a list of bagged trees
def baggingPredict(trees, row):

    predictionByEachTree=list()

    for tree in trees:
        prediction=getLeaf(regression(row, tree))
        predictionByEachTree.append(prediction)

    return predictionByEachTree


# Make the average prediction from list of values predicted by each tree
def avgPrediction(predictionList):

    avgPredValue= int(round(sum(predictionList)/len(predictionList)))
    return avgPredValue
#____________________________________________Calculate Error_____________________________________________________

def calMeanAbsoluteError(absoluteDiff):

    sum=0
    for i in range(0,len(absoluteDiff)):
        sum=sum+absoluteDiff[i]

    modelError=sum/len(absoluteDiff)

    return modelError

#____________________________________________Load and prepare data________________________________________________

# Function to calculate cross validation split for records
def calCrossValidationSplit(data,nFolds):

    #List of list containing folds
    dataSplit = list()
    dataCopy = list(data)
    foldSize = int(len(data) / nFolds)
    for i in range(nFolds):
        fold = list()
        while len(fold) < foldSize:
            index = randrange(len(dataCopy))
            fold.append(dataCopy.pop(index))
        dataSplit.append(fold)
    return dataSplit

# load data
def loadData(filename):
    df = pd.read_csv(filename)
    df_new = df[["DRG Definition", "Region_Numeric_Values1", "Total Discharges",
                 "Median House Price 2011", "Average Total Payments"]]
    dataList = df_new.values.tolist()

    return dataList


#______________________________________main function___________________________________________

def main():

    filename = "Totally_PreProcessedData.csv"
    data = loadData(filename)

    # Setting Hyperparameters

    # Size of each sample from subsampling
    sample_size = 1.0

    # Initial depth
    depth = 1

    # Number of tress
    print("How Many Trees do you want to train your model?:\n")
    nTrees=int(input())

    # Hyper parameter: Depth of the tree
    print("\nWhat should be the maximum depth of the tree?\n")
    maxDepth = int(input())

    # Hyper parameter: number of folds
    print("\nWhat should be the number of folds?\n")
    nFolds = int(input())

    # Hyper parameter: minimum number of samples reached at leaf node
    print("\nWhat should be minimum number of samples reached at leaf node?\n")
    minSamples = int(input())

    # Calling function calCrossValidationSplit to split training set and testing set
    folds = calCrossValidationSplit(data, nFolds)
    for fold in folds:
        trainSet = list(folds)
        trainSet.remove(fold)
        trainSet = sum(trainSet, [])
        testSet = list()
        for record in fold:
            recordCopy = list(record)
            testSet.append(recordCopy)



    #Calling Random Forest model to build trees on training data
    trees=randomForest(trainSet, nTrees, sample_size,maxDepth, minSamples, depth)


    # This list will have absolute difference between actual and predicted values
    absoluteDiff=list()

    #Passing the test data to Random Forest Model
    for row in testSet:
        predictionList=baggingPredict(trees,row)
        print("\nThe list of predictions for a row in test by each tree:\n")
        print("Actual: %s. Predicted: %s" % (row[-1],predictionList))
        avgPredValue=avgPrediction(predictionList)
        print("\nThe final predicted value for row %s by RF trees is %d" % (row[-1],avgPredValue))
        diff=abs(row[-1]-avgPredValue)/row[-1]
        absoluteDiff.append(diff)


    #Calculating mean absolute error of a model
    MAE=calMeanAbsoluteError(absoluteDiff)

    print("\nMean absolute error of a model is:\n")
    print(MAE)





if __name__ == "__main__":
    main()



