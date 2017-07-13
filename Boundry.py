

class Boundry(object):
    """This class represents a character boundry. It has an offset, which is the number of boundries from the beginning of source document where the boundry to the left of the first character is boundry 0. For example, the space between the '"' and the 'T' at the beginning of this doc string would by boundry 0. X is the length of the vector of characters. N is the number of characters preceeding the offset.

    :param offset: The location of this boundry in the document relative to the beginning.
    :param preceeding: The array of characters preceding this boundry.
    :param following: The array of characters following the boundry.
    """
    def __init__(self, offset, preceeding, following):
        self.offset = offset
        self.preceeding = preceeding
        self.following = following

        intRep = []
        if len(preceeding) > 0:
            if type(preceeding[0]) is str:
                for char in preceeding:
                    intRep.append(ord(char))
            elif type(preceeding[0]) is int:
                for integer in preceeding:
                    intRep.append(integer)

        if len(following) > 0:
            if type(following[0]) is str:
                for char in following:
                    intRep.append(ord(char))
            elif type(following[0]) is int:
                for integer in following:
                    intRep.append(integer)
        self.intArray = intRep

    def getX(self):
        return len(self.preceeding) + len(self.following)

    def getN(self):
        return len(self.preceeding)

    def getOffset(self):
        return self.offset

    def getArray(self):
        return self.intArray



class TrainingDatum(Boundry):
    """
    This class represents a single training datum. It extends Boundry by adding an additional property called isNewSentence which has a boolean value indicating whether or not this offset marks the beginning of a new sentence.
    """
    def __init__(self, offset, preceeding, following, isNewSentence):
        super(TrainingDatum, self).__init__(offset, preceeding, following)
        self.isNewSentence = isNewSentence

    def __str__(self):
        return str(self.getArray())
