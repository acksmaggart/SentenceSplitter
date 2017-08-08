from BoundaryReader import parseSentenceBoundaries
from eHostessAddins.SentenceReconstructor import SentenceReconstructor
from eHostessAddins.SentenceRepeatManager import SentenceRepeatManager
from pyConTextNLP.helpers import sentenceSplitter

testDoc = '327000.txt'

inFile = open("./ClinicalNotes/Training1/" + testDoc, 'rU')
body = inFile.read()
inFile.close()
testBoundaries, testText = parseSentenceBoundaries(body)

reconstructor = SentenceReconstructor(testText)
repeatManager = SentenceRepeatManager()

sentences = sentenceSplitter().splitSentences(testText)

predictedBoundaries = []

for sentence in sentences:
    reconstructedSentence = reconstructor.reconstructSentence(sentence)
    docSpan = repeatManager.processSentence(reconstructedSentence, testText)

    predictedBoundaries.append(docSpan[1])

numTrueBoundaries = len(testBoundaries)
numPredictedBoundaries = len(predictedBoundaries)
numPredictedCorrect = 0

for boundary in predictedBoundaries:
    if boundary in testBoundaries:
        numPredictedCorrect += 1

recall = None
if numTrueBoundaries == 0:
    recall = 0.
else:
    recall = float(numPredictedCorrect) / float(numTrueBoundaries)

precision = None
if numPredictedBoundaries == 0:
    precision = 0
else:
    precision = float(numPredictedCorrect) / float(numPredictedBoundaries)

fScore = 2. * recall * precision / (recall + precision)

print "Recall: %f\nPrecision: %f\nF-Score:%f" % (recall, precision, fScore)
