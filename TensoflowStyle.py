import tensorflow as tf
import NoteProcessing as Processor
from datetime import datetime
import numpy as np
import TensorflowUtils

# # create the training data and test data. The training data will be entered as an n by m tensor and the test data as a
# # 1 by n tensor. Use tf.nn.top_k to find the closest points and their indices.
#
# # maximum value for numPreceeding and numFollowing
# maxSideLength = 30
#
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
# trainingData = Processor.parseMultipleTrainingDocs(trainingDocs, sentenceBoundaries, maxSideLength, maxSideLength)
#
# testDocName = "Text1.txt"
# testDocBoundaryArray = text1SentenceBoundries
# inFile = open(testDocName, 'rU')
# testDocContent = inFile.read()
# inFile.close()
#
# results = []
# testData = Processor.parseTestDoc(testDocContent, maxSideLength, maxSideLength)
# testBoundariesVector = np.array(testDocBoundaryArray)
#
# trainingPoints, trainingBoundariesVector = TensorflowUtils.createTrainingArrays(trainingData)
# testPoints = TensorflowUtils.createTestArray(testData)

# Define the tensorflow graph
# Initially only train on threshold weight.

tfMaxSideLength = 3

numNeighbors = 2
tfThresholdWeight = tf.Variable([0.5], "float32")
tfNumPreceeding = tf.constant(2, dtype="int32")
tfNumFollowing = tf.constant(2, dtype="int32")

tfTrainingPoints = tf.placeholder(dtype="float32", shape=[None, tfMaxSideLength * 2])
tfTrainingBoundariesVector = tf.placeholder("float32", [None])
tfTestPoints = tf.placeholder(dtype="float32", shape=[None, tfMaxSideLength * 2])
tfTestBoundariesVector = tf.placeholder("float32", [None])

tfTruncationStart = tfMaxSideLength - tfNumPreceeding


tfNumDimensions = tfNumPreceeding + tfNumFollowing
tfNumTrainingPoints = tf.shape(tfTrainingPoints)[0]
tfTrainingPointsTruncated = tf.transpose(tf.slice(tf.transpose(tfTrainingPoints), [tfTruncationStart, 0], [tfNumDimensions, tfNumTrainingPoints]))

tfNumTestPoints = tf.shape(tfTestPoints)[0]
tfTestPointsTruncated = tf.transpose(tf.slice(tf.transpose(tfTestPoints), [tfTruncationStart, 0], [tfNumDimensions, tfNumTestPoints]))

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
tfLoss = 1 - tfAccuracy


session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

feedDict = {
    tfTrainingPoints: [[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.], [13., 14., 15., 16., 17., 18.]],
    tfTestPoints: [[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.], [14., 15., 16., 17., 18., 19.], [20., 21., 22., 23., 24., 25.]],
    tfTrainingBoundariesVector: [1, 0, 1],
    tfTestBoundariesVector: [1, 0, 0, 0]
}

print session.run([tfTopKIndices, tfTopKValues, tfDistancesWeighted, tfPredictionScore, tfPrediction, tfMatchVector, tfAccuracy], feedDict)

