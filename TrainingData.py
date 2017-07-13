import Boundry
import numpy as np
import numpy.linalg as linalg

class TrainingData:
    def __init__(self, trainingBoundries):
        if not trainingBoundries:
            self.trainingBoundries = []
        else:
            self.trainingBoundries = trainingBoundries

    def addBoundry(self, newBoundry):
        self.trainingBoundries.append(newBoundry)

    def nearestKNeighbors(self, queryBoundry, k, norm):
        """Iterates through all of the training points, calculating the distance from the query point and generating a 2-tuple with the index of the training point and its distance from the query point. It then sorts the list by distance and returns the top k entries. norm is used to determing which order norm to use to calculate distance."""

        if k > len(self.trainingBoundries):
            print "Warning: Oops! There are only %d training data points. You requested %d neighbors. Returning all points."\
                  % (len(self.trainingBoundries), k)
            return self.trainingBoundries

        distances = []
        queryArray = np.array(queryBoundry.getArray())
        for index, trainingPoint in enumerate(self.trainingBoundries):
            trainingArray = np.array(trainingPoint.getArray())
            distance = linalg.norm(trainingArray - queryArray, ord=norm)
            distances.append((index, distance))

        distances.sort(key=lambda distanceTuple: distanceTuple[1])

        neighbors = []
        for iterIndex in range(k):
            neighbors.append(self.trainingBoundries[distances[iterIndex][0]])

        return neighbors

    @classmethod
    def MergeTrainingData(cls, trainingData1, trainingData2):
        trainingDataCombined = []
        for datum in trainingData1.trainingBoundries:
            trainingDataCombined.append(datum)
        for datum in trainingData2.trainingBoundries:
            trainingDataCombined.append(datum)

        return cls(trainingDataCombined)
