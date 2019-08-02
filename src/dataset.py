__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import pandas as pd
import os
import numpy as np

import torch


class Netflix:

    def __init__(self):
        BASEPATH = "../data/netflix"

        self.raw_data: pd.DataFrame = pd.read_csv(os.path.join(BASEPATH, "combined_data_1.txt"), sep=',', header=None,
                                                  names=["userID", "rating", "timestamp"])

        df_nan = pd.DataFrame(pd.isnull(self.raw_data['rating']))
        df_nan = df_nan[df_nan['rating'] == True]
        df_nan = df_nan.reset_index()

        movie_np = []
        movie_id = 1

        for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
            # numpy approach
            temp = np.full((1, i - j - 1), movie_id)
            movie_np = np.append(movie_np, temp)
            movie_id += 1

        # Account for last record and corresponding length
        # numpy approach
        last_record = np.full((1, len(self.raw_data) - df_nan.iloc[-1, 0] - 1), movie_id)
        movie_np = np.append(movie_np, last_record)

        self.raw_data = self.raw_data[pd.notnull(self.raw_data['rating'])]

        self.raw_data['itemID'] = movie_np.astype(int)
        print()

class MovieLens:

    def __init__(self):
        BASEPATH = "../data/ml-100k/ml-100k"

        self.raw_data: pd.DataFrame = pd.read_csv(os.path.join(BASEPATH, "u.data"), sep='\t', header=None,
                                                  names=["userID", "itemID", "rating", "timestamp"])

        self.data: pd.DataFrame = self.raw_data.copy(deep=True)

        self.info = pd.read_csv(os.path.join(BASEPATH, "u.info"), sep='\t', header=None)

        # The last 19 fields are the genres, a 1 indicates the movie
        #               is of that genre, a 0 indicates it is not; movies can be in
        #               several genres at once.
        self.items = pd.read_csv(os.path.join(BASEPATH, "u.item"), sep='|', header=None,
                                 names=["movie id", "movie title", "release date", "video release date",
                                        "IMDb URL", "unknown", "Action", "Adventure", "Animation",
                                        "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                                        "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                                        "Thriller", "War", "Western"], encoding="ISO-8859-1")

        self.items['mean_rating'] = self.items['movie id'].apply(
            lambda itemID: self.data.loc[self.data['itemID'] == itemID, 'rating'].mean())
        self.items['rating_count'] = self.items['movie id'].apply(
            lambda itemID: self.data.loc[self.data['itemID'] == itemID, 'rating'].count())

        self.genres = pd.read_csv(os.path.join(BASEPATH, "u.genre"), sep='|', header=None)

        self.users = pd.read_csv(os.path.join(BASEPATH, "u.user"), sep='|', header=None,
                                 names=["user id", "age", "gender", "occupation", "zip code"])

        self.occupations = pd.read_csv(os.path.join(BASEPATH, "u.occupation"), sep='|', header=None)

    def to_surprise(self):
        return self.data[["userID", "itemID", "rating"]]

    def top_movies(self, n=5):
        top_n_movies_id = self.data[['itemID', 'rating']].groupby(by='itemID').mean().sort_values(by=['rating'],
                                                                                                  ascending=False).index.values[
                          :n]
        return self.items.loc[top_n_movies_id, :]

    def create_cold_start_items(self, n_ratings_threshold):
        self.ccs_threshold = n_ratings_threshold

        rating_count_series = self.data.groupby(by='itemID')['rating'].count()
        self.ccs_itemIDs = rating_count_series.loc[rating_count_series < n_ratings_threshold].index.values
        self.ncs_itemIDs = rating_count_series.loc[rating_count_series >= n_ratings_threshold].index.values

        # clear ccs item ratings
        self.data.loc[self.data['itemID'].isin(self.ccs_itemIDs), 'rating'] = np.nan

        self.data.dropna(inplace=True)

        print(f"CCS item count:{self.ccs_itemIDs.shape}")
        print(f"NCS item count:{self.ncs_itemIDs.shape}")

        return

    def append_rating(self, user, item, rating):
        self.data = self.data.append(other=pd.Series({'userID': user,
                                                      'itemID': item,
                                                      'rating': rating,
                                                      'timestamp': np.nan}), ignore_index=True)

    def rating(self, user, item):
        return self.data.loc[((self.data['userID'] == user) & (self.data['itemID'] == item)), 'rating'].values[0]

    def rating_count(self, item):
        return self.data.loc[self.data['itemID'] == item].shape[0]

    def pick_random_user(self):
        return self.data['userID'].sample(n=1).item()

    def ccs_items(self):  # ccs item generator
        return (itemID for itemID in self.ccs_itemIDs)

    def is_ccs(self, item):
        return np.sum(self.data['itemID'] == item) < self.ccs_threshold

    def rated_ncs_items(self, user):
        return self.data.loc[self.data['userID'] == user, 'itemID'].values

    def features(self, item):
        return self.items.loc[self.items['movie id'] == item].drop(['movie id', 'movie title',
                                                                    'release date',
                                                                    'video release date',
                                                                    'IMDb URL'], axis=1).values

    def __getitem__(self, ix):
        itemID = self.ncs_itemIDs[ix]
        X = self.features(itemID)
        y = 0
        return torch.from_numpy(X.astype(float)), y

    def __len__(self):
        return self.ncs_itemIDs.shape[0]

    def user_item_matrix(self):
        return self.data.pivot_table(values='rating', index='userID', columns='itemID').fillna(0)


if __name__ == "__main__":
    np.random.seed(42)
    # mv = MovieLens()
    mv = Netflix()
    # mv.create_cold_start_items(n_ratings_threshold=50)
    # ccs_item = mv.ccs_items().__next__()
    # print(mv.is_ccs(ccs_item))
    # print(mv.rated_ncs_items(1))
    # mv.pick_random_user()
    # mv.append_rating(-1, -2, -3)
    # print(mv.data.tail())


    # print(mv.data.head())
    # print(mv.info.head())
    # print(mv.items.head())
    # print(mv.genre.head())
    # print(mv.user.head())
    # print(mv.occupation.head())
    # print(mv.top_movies())

    # mv.data.groupby(by='itemID')[['itemID', 'rating']].mean()
    #
    # index = list(mv.data['userID'].unique())
    # columns = list(mv.data['itemID'].unique())
    # index = sorted(index)
    # columns = sorted(columns)
    #
    # util_df = mv.data.pivot_table(values='rating', index='userID', columns='itemID').fillna(0)
    #
    # # item(movie) indices for data types
    # NCS = []
    # CCS = []
    # ICS = []
    # detect_CS = util_df.astype(bool).sum(axis=0)
    # for i, v in detect_CS.iteritems():
    #     if v > 100:
    #         NCS.append(i)
    #     elif v < 5:
    #         CCS.append(i)
    #     else:
    #         ICS.append(i)
    # print(len(NCS))
    # print(len(ICS))
    # print(len(CCS))
    # NCS_train_indices = np.random.rand(len(NCS)) < 0.8
    # ICS_train_indices = np.random.rand(len(ICS)) < 0.8
    # CCS_train_indices = np.random.rand(len(CCS)) < 0.8
    #
    # NCS = np.asarray(NCS)
    # ICS = np.asarray(ICS)
    # CCS = np.asarray(CCS)
    #
    # NCS_train = mv.data.loc[mv.data['item id'].isin(np.asarray(NCS)[NCS_train_indices])]
    # NCS_test = mv.data.loc[mv.data['item id'].isin(np.asarray(NCS)[~NCS_train_indices])]
    # ICS_train = mv.data.loc[mv.data['item id'].isin(np.asarray(ICS)[ICS_train_indices])]
    # ICS_test = mv.data.loc[mv.data['item id'].isin(np.asarray(ICS)[~ICS_train_indices])]
    # CCS_train = mv.data.loc[mv.data['item id'].isin(np.asarray(CCS)[CCS_train_indices])]
    # CCS_test = mv.data.loc[mv.data['item id'].isin(np.asarray(CCS)[~CCS_train_indices])]

    """
    671 unique users whose userid range from 1->671.
    Bir film icin; en fazla:583 en az:1 oy var.
    
    (Pdb) NCS_test.shape
    (14412, 4)
    (Pdb) ICS_test.shape
    (7608, 4)
    (Pdb) CCS_test.shape
    (159, 4)
    (Pdb) NCS_train.shape
    (50007, 4)
    (Pdb) ICS_train.shape
    (27260, 4)
    (Pdb) CCS_train.shape
    (554, 4)
    
    """
