import pandas as pd
import os
import pdb
import numpy as np

class MovieLens:

    def __init__(self):
        BASEPATH = "../data/ml-100k/ml-100k"

        self.data = pd.read_csv(os.path.join(BASEPATH, "u.data"), sep='\t', header=None,
                                names=["user id", "item id", "rating", "timestamp"])

        self.info = pd.read_csv(os.path.join(BASEPATH, "u.info"), sep='\t', header=None)

        # The last 19 fields are the genres, a 1 indicates the movie
        #               is of that genre, a 0 indicates it is not; movies can be in
        #               several genres at once.
        self.items = pd.read_csv(os.path.join(BASEPATH, "u.item"), sep='|', header=None, index_col=0,
                                names=["movie id", "movie title", "release date", "video release date",
                                       "IMDb URL", "unknown", "Action", "Adventure", "Animation",
                                       "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                                       "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                                       "Thriller", "War", "Western"], encoding="ISO-8859-1")

        self.genres = pd.read_csv(os.path.join(BASEPATH, "u.genre"), sep='|', header=None)

        self.users = pd.read_csv(os.path.join(BASEPATH, "u.user"), sep='|', header=None,
                                names=["user id", "age", "gender", "occupation", "zip code"])

        self.occupations = pd.read_csv(os.path.join(BASEPATH, "u.occupation"), sep='|', header=None)

    def top_movies(self, n=5):
        top_n_movies_id = self.data[['item id', 'rating']].groupby(by='item id').mean().sort_values(by=['rating'], ascending=False).index.values[:n]
        return self.items.loc[top_n_movies_id, :]

mv = MovieLens()
#print(mv.data.head())
# print(mv.info.head())
#print(mv.items.head())
# print(mv.genre.head())
# print(mv.user.head())
# print(mv.occupation.head())
#print(mv.top_movies())

index=list(mv.data['user id'].unique())
columns=list(mv.data['item id'].unique())
index=sorted(index)
columns=sorted(columns)
 
util_df=mv.data.pivot_table(values='rating',index='user id',columns='item id')
util_df = util_df.fillna(0)

#item(movie) indices for data types
NCS = []
CCS = []
ICS = []
detect_CS = util_df.astype(bool).sum(axis=0)
for i,v in detect_CS.iteritems():
    if v>100:
        NCS.append(i)		
    elif v<5:
        CCS.append(i)		
    else:
        ICS.append(i)		
print(len(NCS))
print(len(ICS))
print(len(CCS))
NCS_train_indices = np.random.rand(len(NCS)) < 0.8
ICS_train_indices = np.random.rand(len(ICS)) < 0.8
CCS_train_indices = np.random.rand(len(CCS)) < 0.8

NCS = np.asarray(NCS)
ICS = np.asarray(ICS)
CCS = np.asarray(CCS)

NCS_train = mv.data.loc[mv.data['item id'].isin(np.asarray(NCS)[NCS_train_indices])]
NCS_test = mv.data.loc[mv.data['item id'].isin(np.asarray(NCS)[~NCS_train_indices])]
ICS_train = mv.data.loc[mv.data['item id'].isin(np.asarray(ICS)[ICS_train_indices])]
ICS_test = mv.data.loc[mv.data['item id'].isin(np.asarray(ICS)[~ICS_train_indices])]
CCS_train = mv.data.loc[mv.data['item id'].isin(np.asarray(CCS)[CCS_train_indices])]
CCS_test = mv.data.loc[mv.data['item id'].isin(np.asarray(CCS)[~CCS_train_indices])]


pdb.set_trace()

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
