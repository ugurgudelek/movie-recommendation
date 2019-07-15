__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"


"""
1. Create MovieLens Dataset
2. Split CCS and NCS items (n_ratings < 50)
3. train AE with movie contents
4. m = pick a CCS item
   loop until m is not CCS anymore:
        u = pick random user
        cos_sims = cosine similarities between latent(u.rated_movies) vs latent(m).
       (u,m).rating = mean(cos_sims.top10())    -> fill movielens dataset accordingly.
5.  Repeat step 4 until 0 CCS item left.
6. Send user x movies utility matrix to surpriselib timeSVD++
"""

from surprise import SVD, SVDpp
from surprise import Dataset
from surprise import accuracy
from surprise import Reader, AlgoBase
from surprise.model_selection import train_test_split

from dataset import MovieLens
from autoencoder import AutoEncoder
from collaborative_filtering import MeanAlgorithm


import numpy as np

mv = MovieLens()
ae = AutoEncoder(input_dim=100)

mv.create_cold_start_items(n_ratings_threshold=50)

ae.train(mv.dataloader)

for ccs_item in mv.ccs_items().random_shuffle():
    while ccs_item.is_ccs():
        u = mv.pick_random_user()
        u_rated_latents = [ae.encode(m.features())for m in u.rated_ncs_movies()]
        ccs_latent = ae.encode(ccs_item.features())

        cosine_sims = [cos(r_latent, ccs_latent) for r_latent in u_rated_latents]

        top10_idxs = np.argsort(cosine_sims)[:10]
        top10_ncs_items = u.rated_ncs_movies()[top10_idxs]
        mean_top10 = np.mean(top10_ncs_items)

        mv.data.append((user=u, item=ccs_item, rating=mean_top10))


data = Dataset.load_from_df(mv.to_surprise(), Reader(rating_scale=(1, 5)))
trainset, testset = train_test_split(data, test_size=.25)

# algo = SVD()
# algo = SVDpp()
algo = MeanAlgorithm()

algo.fit(trainset)
predictions = algo.test(testset, verbose=False)

accuracy.rmse(predictions)