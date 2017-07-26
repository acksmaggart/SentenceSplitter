import re

def parseSentenceBoundaries(documentBodyWithBoundaryMarkers, sepChar='^'):
    """
    Parses documents and returns the location of the sentence boundary markers along with the clean document string.
    :param documentBodyWithBoundaryMarkers: The text with sepChars marking sentence boundaries.
    :return: Location of the boundary markers along with the clean text.
    """

    offsetCount = 0
    boundaries = []
    for char in documentBodyWithBoundaryMarkers:
        if char == sepChar:
            boundaries.append(offsetCount)
            continue
        offsetCount += 1

    cleanText = re.sub(re.escape(sepChar), "", documentBodyWithBoundaryMarkers)

    return boundaries, cleanText