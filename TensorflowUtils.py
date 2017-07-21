import numpy as np

def createTrainingArrays(trainingData, numPreceeding, numFollowing):
    """
    Takes the training data and returns the appropriate numpy arrays to feed to tensorflow. All of the training-data
    points are combined into a single matrix and the sentence-boundary indicies are combined into a single matrix
    ensuring that the value of the index reflects the training datum's position in the training-point array.
    :param trainingData:
    :return:
    """

    numTrainingPoints = len(trainingData.trainingBoundries)
    pointDimension = numPreceeding + numFollowing

    trainingPointArray = np.zeros((numTrainingPoints, pointDimension))
    boundaryList = []

    for index, sentenceBoundary in enumerate(trainingData.trainingBoundries):
        trainingPointArray[index] = np.array(sentenceBoundary.getArray())
        if sentenceBoundary.isNewSentence:
            boundaryList.append(index)

    boundaryNpArray = np.array(boundaryList)

    return trainingPointArray, boundaryNpArray

def createTestArray(testBoundaries, numPreceeding, numFollowing):

    boundaryArray = np.zeros((len(testBoundaries), numPreceeding + numFollowing))

    for index, boundary in testBoundaries:
        boundaryArray[index] = np.array(boundary.getArray())

    return boundaryArray
