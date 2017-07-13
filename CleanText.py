import re

inputFile = 'Text5.txt'

inFileHandle = open(inputFile, 'r')
contents = inFileHandle.read()
inFileHandle.close()

for index, char in enumerate(contents):
    if ord(char) > 127:
        print "%s, %d" % (char, index)


