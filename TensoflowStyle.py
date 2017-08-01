import tensorflow as tf
import NoteProcessing as Processor
from datetime import datetime
import sys
import os
import TensorflowUtils
from BoundaryReader import parseSentenceBoundaries

# maximum value for numPreceeding and numFollowing
print "Starting Program"
maxSideLength = 30

# text1SentenceBoundries = [255, 374, 477, 627, 821, 918, 1058, 1249, 1299, 1494, 1697,
#                           1874, 2151, 2279, 2378, 2486, 2628, 2773, 2976, 3089, 3264,
#                           3518, 3676, 3778, 3870, 4008, 4079]
#
# text2SentenceBoundries = [101, 155, 243, 501, 760, 1010, 1035, 1059, 1149, 1219, 1229,
#                           1373, 1474, 1550, 1608, 1675, 1795, 1872, 2096, 2222, 2276,
#                           2454, 2541, 2639, 2771, 2995, 3166, 3236, 3375, 3437, 3549,
#                           3605, 3670, 3776, 3884, 4002, 4038, 4044, 4166, 4216, 4359,
#                           4486, 4576, 4647, 4657, 4769, 4847]
#
# text3SentenceBoundries = [150, 417, 603, 848, 1083, 1366, 1513, 1611, 1885, 1985, 2019,
#                            2052, 2158, 2230, 2421, 2559, 2682, 2901, 3001]
#
# text4SentenceBoundries = [212, 352, 607, 790, 858, 1103, 1189, 1258, 1512, 1647, 1818, 1887,
#                           2007, 2208, 2351, 2429, 2568, 2759]
#
# text5SentenceBoundries = [112, 184, 379, 649, 706, 889, 948, 1161, 1229, 1430, 1534, 1566,
#                           1722, 1800, 1890, 2064, 2148, 2232, 2337, 2737, 2775, 2821, 2926,
#                           3055, 3143, 3269, 3461, 3493, 3647, 3679, 3872, 4039, 4094, 4132,
#                           4184, 4230, 4289, 4341, 4690, 4734, 4774, 4846, 4897, 5067, 5242,
#                           5506, 5585, 5781, 5837, 6051, 6144, 6291, 6375, 6481, 6514, 6703,
#                           6845, 6875, 7079, 7126, 7178, 7250, 7439, 7457, 7524, 7544, 7600,
#                           7776, 7831, 7866, 7881, 8055, 8142, 8289]
#
# trainingDocs = ['Text2.txt', 'Text3.txt', 'Text4.txt', 'Text5.txt']
# sentenceBoundaries = [text2SentenceBoundries, text3SentenceBoundries, text4SentenceBoundries, text5SentenceBoundries]
#
# print "About to create training data."
# trainingData = Processor.parseMultipleTrainingDocs(trainingDocs, sentenceBoundaries, maxSideLength, maxSideLength)
# print "Training data created."
#
# testDocName = "Text1.txt"
# testDocBoundaryArray = text1SentenceBoundries
# inFile = open(testDocName, 'rU')
# testDocContent = inFile.read()
# inFile.close()
#
# print "About to create test data."
# testData = Processor.parseTrainingDoc(testDocContent, text1SentenceBoundries, maxSideLength, maxSideLength)
# print "Test data created."
#
# print "About to create npArrays."
# trainingPoints, trainingBoundariesVector = TensorflowUtils.createTrainingArrays(trainingData, maxSideLength, maxSideLength)
# testPoints, testBoundariesVector = TensorflowUtils.createTrainingArrays(testData, maxSideLength, maxSideLength)
# print "npArrays created."


trainingDocs = ['327000.txt', '10020.txt', '457663.txt']
testDoc = '10018.txt'

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

print "About to create training data."
trainingData = Processor.parseMultipleTrainingDocStrings(trainingDocs, trainingBoundaries, maxSideLength, maxSideLength)
print "Training data created."


print "About to create test data."
testData = Processor.parseSingleTrainingDocString(testText, testBoundaries, maxSideLength, maxSideLength)
print "Test data created."

print "About to create npArrays."
trainingPoints, trainingBoundariesVector = TensorflowUtils.createTrainingArrays(trainingData, maxSideLength, maxSideLength)
testPoints, testBoundariesVector = TensorflowUtils.createTrainingArrays(testData, maxSideLength, maxSideLength)
print "npArrays created."

# Define the tensorflow graph
# Initially only train on threshold weight.

print "Defining tensorflow graph."
tfMaxSideLength = maxSideLength

numNeighbors = tf.placeholder("int32")
tfThresholdWeight = tf.placeholder("float32")
tfNumPreceeding = tf.placeholder( dtype="int32")
tfNumFollowing = tf.placeholder( dtype="int32")

tfTrainingPoints = tf.placeholder(dtype="float32", shape=[None, tfMaxSideLength * 2])
tfTrainingBoundariesVector = tf.placeholder("float32", [None])
tfTestPoints = tf.placeholder(dtype="float32", shape=[None, tfMaxSideLength * 2])
tfTestBoundariesVector = tf.placeholder("float32", [None])

tfTruncationStart = tfMaxSideLength - tfNumPreceeding


tfNumDimensions = tfNumPreceeding + tfNumFollowing
tfNumTrainingPoints = tf.shape(tfTrainingPoints)[0]
tfTrainingPointsTruncated = tf.transpose(tf.slice(tf.transpose(tfTrainingPoints), [tfTruncationStart, 0], [tfNumDimensions, tfNumTrainingPoints]))
tfTrainingPointsTruncatedShape = tf.shape(tfTrainingPointsTruncated)

tfNumTestPoints = tf.shape(tfTestPoints)[0]
tfTestPointsTruncated = tf.transpose(tf.slice(tf.transpose(tfTestPoints), [tfTruncationStart, 0], [tfNumDimensions, tfNumTestPoints]))
tfTestPointsTruncatedShape = tf.shape(tfTestPointsTruncated)

tfDistance = tf.reduce_sum(tf.abs(tf.subtract(tfTestPointsTruncated, tf.expand_dims(tfTrainingPointsTruncated, 1))), axis=2)

tfTopKValuesNegative, tfTopKIndices = tf.nn.top_k(tf.transpose(tf.negative(tfDistance)), k=numNeighbors)
tfTopKValues = tf.negative(tfTopKValuesNegative)

# Normalize the distances, invert them, and multiply them by 1 or 0 depending on whether or not they are a sentence boundary.
tfDistancesWeighted = 1 - tf.divide(tfTopKValues, tf.expand_dims(tf.reduce_sum(tfTopKValues, axis=1), 1))
tfPredictionScore = tf.reduce_sum(tf.multiply(tf.gather(tfTrainingBoundariesVector, tfTopKIndices), tfDistancesWeighted), axis=1)

# Evaluate accuracy.
tfPrediction = tf.where(tf.greater_equal(tfPredictionScore, tfThresholdWeight), tf.ones(tf.shape(tfTestBoundariesVector), dtype="float32"), tf.zeros(tf.shape(tfTestBoundariesVector), dtype="float32"))
tfMatchVector = tf.where(tf.equal(tfPrediction, tfTestBoundariesVector), tf.ones(tf.shape(tfTestBoundariesVector), dtype="float32"), tf.zeros(tf.shape(tfTestBoundariesVector), dtype="float32"))
tfAccuracy = tf.div(tf.reduce_sum(tfMatchVector), tf.cast(tf.shape(tfMatchVector), tf.float32))


#  numPreceedingArray = [0, 1, 3, 5, 10, 20]
# numFollowingArray = [0, 1, 3, 5, 10, 20]
# numNeighborsArray = [1, 3, 5, 7]
# thresholdArray = [0.1, 0.3, 0.5, 0.7, 0.9, 0.999]
numPreceedingArray = [1]
numFollowingArray = [0]
numNeighborsArray = [1]
thresholdArray = [0.1]

start = datetime.now()

iterationCount = 0
totalIterations = len(numPreceedingArray) * len(numFollowingArray) * len(numNeighborsArray) * len(thresholdArray)

outFilePath = "Report.txt"
if os.path.isfile(outFilePath):
    os.remove(outFilePath)
outFile = open(outFilePath, 'a')
outFile.write("NumPreceeding\tNumFollowing\tNumNeighbors\tThreshold\tAccuracy\tRuntime\n")

print "About to start iterations."
for numPreceedingIteration in numPreceedingArray:
    for numFollowingIteration in numFollowingArray:
        for numNeighborsIteration in numNeighborsArray:
            for thresholdIteration in thresholdArray:

                if numPreceedingIteration == 0 and numFollowingIteration == 0:
                    continue

                singleRunStart = datetime.now()
                session = tf.Session()

                feedDict = {
                    tfTrainingPoints: trainingPoints,
                    tfTestPoints: testPoints,
                    tfTrainingBoundariesVector: trainingBoundariesVector,
                    tfTestBoundariesVector: testBoundariesVector,
                    tfNumPreceeding: numPreceedingIteration,
                    tfNumFollowing: numFollowingIteration,
                    tfThresholdWeight: thresholdIteration,
                    numNeighbors: numNeighborsIteration
                }

                accuracyResult, numPreceedingResult, numFollowingResult, numNeighborsResult, thresholdResult, shapeTest, shapeTraining = session.run([tfAccuracy, tfNumPreceeding, tfNumFollowing, numNeighbors, tfThresholdWeight, tfTestPointsTruncatedShape, tfTrainingPointsTruncatedShape], feedDict)
                print shapeTest
                print shapeTraining
                # if accuracyResult[0] > best["accuracy"]:
                #     best["accuracy"] = accuracyResult[0]
                #     best["numPreceeding"] = numPreceedingResult,
                #     best["numFollowing"] = numFollowingResult
                #     best["numNeighbors"] = numNeighborsResult
                #     best["threshold"] = thresholdResult
                #
                # if accuracyResult[0] < worst["accuracy"]:
                #     worst["accuracy"] = accuracyResult[0]
                #     worst["numPreceeding"] = numPreceedingResult,
                #     worst["numFollowing"] = numFollowingResult
                #     worst["numNeighbors"] = numNeighborsResult
                #     worst["threshold"] = thresholdResult

                session.close()
                singleRunEnd = datetime.now()
                singleRunElapsed = singleRunEnd - singleRunStart
                outString = "%d\t%d\t%d\t%.3f\t%f\t%s\n" % (numPreceedingResult, numFollowingResult, numNeighborsResult, thresholdResult, accuracyResult[0], str(singleRunElapsed))
                outFile.write(outString)
                outFile.flush()

                iterationCount += 1
                sys.stdout.write("\rCompleted iteration %d of %d. (%.2f%%)" % (iterationCount, totalIterations, (float(iterationCount)/float(totalIterations))*100.))

outFile.close()
end = datetime.now()
print ""
print "Elapsed: %s" % (end - start)
# print ""
# print "Best:"
# print best
# print ""
# print ""
# print "Worst:"
# print worst
