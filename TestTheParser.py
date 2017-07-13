import NoteProcessing as Processor

# inFile = open('./Text1.txt', 'r')
# doc = inFile.read()
# inFile.close()
# testDoc = '1234567890'
#
# trainingData = Processor.parseTrainingDoc(doc, [2, 5], 2, 4)
#
# for trainingDatum in trainingData.trainingBoundries:
#     print trainingDatum.preceeding + '|' + trainingDatum.following + ' ' + str(trainingDatum.isNewSentence)

text1SentenceBoundries = [255, 374, 477, 627, 821, 918, 1058, 1249, 1299, 1494, 1697,
                          1874, 2151, 2279, 2378, 2486, 2628, 2773, 2976, 3089, 3264,
                          3518, 3676, 3778, 3870, 4008, 4079]

text2SentenceBoundries = [101, 155, 243, 501, 760, 1010, 1035, 1059, 1149, 1219, 1229,
                          1373, 1474, 1550, 1608, 1675, 1795, 1872, 2096, 2222, 2276,
                          2454, 2541, 2639, 2771, 2995, 3166, 3236, 3375, 3437, 3549,
                          3605, 3670, 3776, 3884, 4002, 4038, 4044, 4166, 4216, 4359,
                          4486, 4576, 4647, 4657, 4769, 4847]

text3SentenceBoundries = [150, 417, 603, 848, 1083, 1366, 1513, 1611, 1885, 1985, 2019,
                           2052, 2158, 2230, 2421, 2559, 2682, 2901, 3001]

text4SentenceBoundries = [212, 352, 607, 790, 858, 1103, 1189, 1258, 1512, 1647, 1818, 1887,
                          2007, 2208, 2351, 2429, 2568, 2759]

text5SentenceBoundries = [112, 184, 379, 649, 706, 889, 948, 1161, 1229, 1430, 1534, 1566,
                          1722, 1800, 1890, 2064, 2148, 2232, 2337, 2737, 2775, 2821, 2926,
                          3055, 3143, 3269, 3461, 3493, 3647, 3679, 3872, 4039, 4094, 4132,
                          4184, 4230, 4289, 4341, 4690, 4734, 4774, 4846, 4897, 5067, 5242,
                          5506, 5585, 5781, 5837, 6051, 6144, 6291, 6375, 6481, 6514, 6703,
                          6845, 6875, 7079, 7126, 7178, 7250, 7439, 7457, 7524, 7544, 7600,
                          7776, 7831, 7866, 7881, 8055, 8142, 8289]

trainingData = Processor.parseMultipleTrainingDocs(['Text5.txt'],
                                                   [text5SentenceBoundries],
                                                   4, 3)

for trainingDatum in trainingData.trainingBoundries:
    print trainingDatum.preceeding + '|' + trainingDatum.following + ' ' + str(trainingDatum.isNewSentence)

# inFile = open('')
# testDoc = ''
# testBoundries = Processor.parseTestDoc()