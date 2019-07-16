__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from surprise import SVD, SVDpp
from surprise import Dataset
from surprise import accuracy
from surprise import Reader, AlgoBase
from surprise.model_selection import train_test_split

from dataset import MovieLens

import numpy as np

class MeanAlgorithm(AlgoBase):

    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

    def fit(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)

        # Compute the average rating. We might as well use the
        # trainset.global_mean attribute ;)
        # self.the_mean = np.mean([r for (_, _, r) in
        #                          self.trainset.all_ratings()])

        self.the_mean = trainset.global_mean

        return self

    def estimate(self, u, i):

        return self.the_mean


class AEAlgorithm(AlgoBase):

    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        return self

    def estimate(self, u, i):
        r = self.trainset.global_mean
        div = 1

        if self.trainset.knows_user(u):
            r += u
            div += 1
        if self.trainset.knows_item(i):
            r += i
            div += 1

        return r / div



if __name__=="__main__":
    np.random.seed(42)

    # The columns must correspond to user id, item id and ratings (in that order).
    # A reader is still needed but only the rating_scale param is requiered.
    data = Dataset.load_from_df(MovieLens().to_surprise(), Reader(rating_scale=(1, 5)))

    # sample random trainset and testset
    # test set is made of 25% of the ratings.
    trainset, testset = train_test_split(data, test_size=.25)

    # We'll use the famous SVD algorithm.
    algo = SVD()
    # algo = SVDpp()
    # algo = MeanAlgorithm()
    # algo = AEAlgorithm()

    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)
    predictions = algo.test(testset, verbose=False)

    # Then compute RMSE
    accuracy.rmse(predictions)