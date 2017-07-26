import re
from BoundaryReader import parseSentenceBoundaries

# inputFile = 'Text5.txt'
#
# inFileHandle = open(inputFile, 'r')
# contents = inFileHandle.read()
# inFileHandle.close()
#
# for index, char in enumerate(contents):
#     if ord(char) > 127:
#         print "%s, %d" % (char, index)



inFile = open('./TestText1.txt')
docBody = inFile.read()
inFile.close()

boundaries, cleanText = parseSentenceBoundaries(docBody)

print "hie!"

