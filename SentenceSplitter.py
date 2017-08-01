import tensorflow as tf
from BoundaryReader import parseSentenceBoundaries
import NoteProcessing as Processor
import TensorflowUtils
from datetime import datetime
import numpy

def detectSentenceBoundaries(docString):
    maxSideLength = 30
    numPreceedingConfig = 0
    numFollowingConfig = 1
    numNeighborsConfig = 1
    thresholdConfig = 0.5


    #Read the representative texts.
    trainingDocs = ['327000.txt', '10020.txt', '457663.txt']

    trainingBoundaries = []
    trainingTexts = []
    for docName in trainingDocs:
        with open("./ClinicalNotes/" + docName, 'rU') as inFile:
            body = inFile.read()
            boundaries, body = parseSentenceBoundaries(body)
            trainingBoundaries.append(boundaries)
            trainingTexts.append(body)

    inFile = open("./ClinicalNotes/" + testDoc, 'rU')
    body = inFile.read()
    inFile.close()
    testBoundaries, testText = parseSentenceBoundaries(body)

    trainingData = Processor.parseMultipleTrainingDocStrings(trainingDocs, trainingBoundaries, maxSideLength,
                                                             maxSideLength)
    testData = Processor.parseSingleTrainingDocString(testText, testBoundaries, maxSideLength, maxSideLength)

    # Convert the data into Numpy Arrays.
    trainingPoints, trainingBoundariesVector = TensorflowUtils.createTrainingArrays(trainingData, maxSideLength,maxSideLength)
    testPoints, testBoundariesVector = TensorflowUtils.createTrainingArrays(testData, maxSideLength, maxSideLength)


    tfMaxSideLength = maxSideLength

    numNeighbors = tf.placeholder("int32")
    tfThresholdWeight = tf.placeholder("float32")
    tfNumPreceeding = tf.placeholder(dtype="int32")
    tfNumFollowing = tf.placeholder(dtype="int32")

    tfTrainingPoints = tf.placeholder(dtype="float32", shape=[None, tfMaxSideLength * 2])
    tfTrainingBoundariesVector = tf.placeholder("float32", [None])
    tfTestPoints = tf.placeholder(dtype="float32", shape=[None, tfMaxSideLength * 2])
    tfTestBoundariesVector = tf.placeholder("float32", [None])

    tfTruncationStart = tfMaxSideLength - tfNumPreceeding

    tfNumDimensions = tfNumPreceeding + tfNumFollowing
    tfNumTrainingPoints = tf.shape(tfTrainingPoints)[0]
    tfTrainingPointsTruncated = tf.transpose(
        tf.slice(tf.transpose(tfTrainingPoints), [tfTruncationStart, 0], [tfNumDimensions, tfNumTrainingPoints]))
    tfTrainingPointsTruncatedShape = tf.shape(tfTrainingPointsTruncated)

    tfNumTestPoints = tf.shape(tfTestPoints)[0]
    tfTestPointsTruncated = tf.transpose(
        tf.slice(tf.transpose(tfTestPoints), [tfTruncationStart, 0], [tfNumDimensions, tfNumTestPoints]))
    tfTestPointsTruncatedShape = tf.shape(tfTestPointsTruncated)

    tfDistance = tf.reduce_sum(tf.abs(tf.subtract(tfTestPointsTruncated, tf.expand_dims(tfTrainingPointsTruncated, 1))),
                               axis=2)

    tfTopKValuesNegative, tfTopKIndices = tf.nn.top_k(tf.transpose(tf.negative(tfDistance)), k=numNeighbors)
    tfTopKValues = tf.negative(tfTopKValuesNegative)

    # Normalize the distances, invert them, and multiply them by 1 or 0 depending on whether or not they are a sentence boundary.
    tfDistancesWeighted = 1 - tf.divide(tfTopKValues, tf.expand_dims(tf.reduce_sum(tfTopKValues, axis=1), 1))
    tfPredictionScore = tf.reduce_sum(
        tf.multiply(tf.gather(tfTrainingBoundariesVector, tfTopKIndices), tfDistancesWeighted), axis=1)

    # Evaluate accuracy.
    # tfPredictions = tf.where(tf.greater_equal(tfPredictionScore, tfThresholdWeight))
    tfPredictions = tf.where(tf.greater_equal(tfPredictionScore, tfThresholdWeight))

    tfPrediction = tf.where(tf.greater_equal(tfPredictionScore, tfThresholdWeight),
                            tf.ones(tf.shape(tfTestBoundariesVector), dtype="float32"),
                            tf.zeros(tf.shape(tfTestBoundariesVector), dtype="float32"))
    tfMatchVector = tf.where(tf.equal(tfPrediction, tfTestBoundariesVector),
                             tf.ones(tf.shape(tfTestBoundariesVector), dtype="float32"),
                             tf.zeros(tf.shape(tfTestBoundariesVector), dtype="float32"))
    tfAccuracy = tf.div(tf.reduce_sum(tfMatchVector), tf.cast(tf.shape(tfMatchVector), tf.float32))


    start = datetime.now()


    session = tf.Session()

    feedDict = {
        tfTrainingPoints: trainingPoints,
        tfTestPoints: testPoints,
        tfTrainingBoundariesVector: trainingBoundariesVector,
        tfTestBoundariesVector: testBoundariesVector,
        tfNumPreceeding: numPreceedingConfig,
        tfNumFollowing: numFollowingConfig,
        tfThresholdWeight: thresholdConfig,
        numNeighbors: numNeighborsConfig
    }

    predictions, accuracy, prediction, testBoundaries = session.run([tfPredictions, tfAccuracy, tfPrediction, tfTestBoundariesVector], feedDict)

    session.close()
    end = datetime.now()

    print ""
    print "Elapsed: %s" % (end - start)

    return predictions, accuracy, prediction, testBoundaries

testDoc = '10018.txt'
inFile = open("./ClinicalNotes/" + testDoc, 'rU')
body = inFile.read()
inFile.close()
testBoundaries, testText = parseSentenceBoundaries(body)

predictions, accuracy, prediction, testBoundaries = detectSentenceBoundaries(testText)


numpy.set_printoptions(threshold=numpy.nan)
print testBoundaries
print ""
print ""
print predictions
print ""
print ""
print accuracy
print prediction
print ""
print ""
print testBoundaries



