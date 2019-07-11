import pandas as pd
import os


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
# print(mv.data.head())
# print(mv.info.head())
print(mv.items.head())
# print(mv.genre.head())
# print(mv.user.head())
# print(mv.occupation.head())

print(mv.top_movies())
