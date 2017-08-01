from Boundry import Boundry
from Boundry import TrainingDatum
from TrainingData import TrainingData
import math


def parseTrainingDoc(doc, sentenceBoundaries, numPreceeding, numFollowing):
    """This function takes a list of documents to process and returns a training data set."""
    trainingData = []
    docLength = len(doc)
    # we need to go all the way to the boundry past the last character, which is at len(doc) + 1
    for index in range(docLength + 1):
        preceeding = []
        following = []
        if index - numPreceeding < 0:
            preceeding = grabPreceedingOverlapsBeginning(doc, index, numPreceeding)
        else:
            preceeding = grabPreceedingNormal(doc, index, numPreceeding)

        if index + numFollowing > docLength:
            following = grabFollowingOverlapsEnding(doc, index, numFollowing)
        else:
            following = grabFollowingNormal(doc, index, numFollowing)

        isSentenceBoundry = False
        if index in sentenceBoundaries:
            isSentenceBoundry = True
        newTrainingDatum = TrainingDatum(index, preceeding, following, isSentenceBoundry)
        trainingData.append(newTrainingDatum)

    return TrainingData(trainingData)

def parseTestDoc(doc, numPreceeding, numFollowing):
    """This function takes a list of documents to process and returns a list of boundary objects."""
    boundaries = []
    docLength = len(doc)
    # we need to go all the way to the boundry past the last character, which is at len(doc) + 1
    for index in range(docLength + 1):
        preceeding = []
        following = []
        if index - numPreceeding < 0:
            preceeding = grabPreceedingOverlapsBeginning(doc, index, numPreceeding)
        else:
            preceeding = grabPreceedingNormal(doc, index, numPreceeding)

        if index + numFollowing > docLength:
            following = grabFollowingOverlapsEnding(doc, index, numFollowing)
        else:
            following = grabFollowingNormal(doc, index, numFollowing)

        newBoundary = Boundry(index, preceeding, following)
        boundaries.append(newBoundary)

    return boundaries


def parseMultipleTrainingDocs(docList, sentenceBoundaries, numPreceeding, numFollowing):
    if len(docList) == 0 or len(sentenceBoundaries) == 0 or len(docList) != len(sentenceBoundaries):
        raise RuntimeError("The length of docList and sentenceBoundaries must be the same non-zero value.")
    doc = ''
    with open(docList[0], 'rU') as inFile:
        doc = inFile.read()
    trainingDataOriginal = parseTrainingDoc(doc, sentenceBoundaries[0], numPreceeding, numFollowing)

    for index in range(1, len(docList)):
        with open(docList[index], 'rU') as inFile:
            doc = inFile.read()
        newTrainingData = parseTrainingDoc(doc, sentenceBoundaries[index], numPreceeding, numFollowing)
        trainingDataOriginal = TrainingData.MergeTrainingData(trainingDataOriginal, newTrainingData)

    return trainingDataOriginal

def parseMultipleTrainingDocStrings(docList, sentenceBoundaries, numPreceeding, numFollowing):
    if len(docList) == 0 or len(sentenceBoundaries) == 0 or len(docList) != len(sentenceBoundaries):
        raise RuntimeError("The length of docList and sentenceBoundaries must be the same non-zero value.")

    trainingDataOriginal = parseTrainingDoc(docList[0], sentenceBoundaries[0], numPreceeding, numFollowing)

    for index in range(1, len(docList)):
        newTrainingData = parseTrainingDoc(docList[index], sentenceBoundaries[index], numPreceeding, numFollowing)
        trainingDataOriginal = TrainingData.MergeTrainingData(trainingDataOriginal, newTrainingData)

    return trainingDataOriginal


def parseSingleTrainingDocString(doc, sentenceBoundaries, numPreceeding, numFollowing):

    return parseTrainingDoc(doc, sentenceBoundaries, numPreceeding, numFollowing)


def grabPreceedingOverlapsBeginning(doc, cursor, numPreceeding):
    difference = int(math.sqrt(math.pow(cursor - numPreceeding, 2)))
    preceeding = ''
    for index in range(difference):
        preceeding += chr(0)

    for index in range(cursor):
        preceeding += doc[index]

    return preceeding

def grabFollowingOverlapsEnding(doc, cursor, numFollowing):
    preceeding = doc[cursor:]

    difference = numFollowing - len(preceeding)

    for _ in range(difference):
        preceeding += chr(0)

    return preceeding

def grabPreceedingNormal(doc, cursor, numPreceeding):
    preceeding = ''
    # range() is used to account for cases where numPreceeding == 0.
    for index in range(cursor - numPreceeding, cursor):
        preceeding += doc[index]
    return preceeding

def grabFollowingNormal(doc, cursor, numFollowing):
    following = ''
    for index in range(cursor, cursor + numFollowing):
        following += doc[index]

    return following




