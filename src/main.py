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

import torch
from torch import nn
from torch.utils.data import DataLoader

from surprise import SVD, SVDpp, KNNBasic
from surprise import Dataset
from surprise import accuracy
from surprise import Reader, AlgoBase
from surprise.model_selection import train_test_split

from dataset import MovieLens
from autoencoder import AutoEncoder
from collaborative_filtering import MeanAlgorithm


import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm
import pandas as pd
import os

OUTPUT_PATH = '../output'
os.makedirs(OUTPUT_PATH, exist_ok=True)

num_epochs = 100
batch_size = 20
learning_rate = 1e-3

# mv = MovieLens()
# mv.create_cold_start_items(n_ratings_threshold=5)
#
# dataloader = DataLoader(mv, batch_size=batch_size, shuffle=True, drop_last=True)
#
# model = AutoEncoder(input_dim=21, latent_dim=5)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# AutoEncoder.fit(model, num_epochs, dataloader, criterion, optimizer)
#
#
# for ccs_item in tqdm(mv.ccs_items()):
#     print('ccs item:', ccs_item)
#     while mv.is_ccs(ccs_item):
#         u = mv.pick_random_user()
#         print('user:', u)
#         rated_ncs_items_by_u = mv.rated_ncs_items(u)
#         u_rated_latents = [model.encode(mv.features(m)) for m in rated_ncs_items_by_u]
#         ccs_latent = model.encode(mv.features(ccs_item))
#
#         cosine_sims = [cosine(r_latent, ccs_latent) for r_latent in u_rated_latents]
#
#         sorted_idxs = np.argsort(cosine_sims)
#         top10_ncs_items = rated_ncs_items_by_u[sorted_idxs][:10]
#         top10_ncs_ratings = [mv.rating(u, m) for m in top10_ncs_items]
#         mean_top10 = np.mean(top10_ncs_ratings)
#
#         mv.append_rating(user=u, item=ccs_item, rating=mean_top10)
#
#         # print(f"\nccs_item:{ccs_item} | user:{u} | rating:{mean_top10} | rcount:{mv.rating_count(ccs_item)}")
#
# mv.data.to_csv('export_data.csv')


ae_data = Dataset.load_from_df(pd.read_csv('export_data.csv', index_col=0)[['userID', 'itemID', 'rating']], Reader(rating_scale=(1, 5)))
raw_data = Dataset.load_from_df(MovieLens().to_surprise(), Reader(rating_scale=(1, 5)))

def run(data):
    trainset, testset = train_test_split(data, test_size=.25)
    # algo = SVD()
    # algo = SVDpp()
    # algo = MeanAlgorithm()
    algo = KNNBasic()
    algo.fit(trainset)
    predictions = algo.test(testset, verbose=False)

    acc = accuracy.rmse(predictions)
    return acc

ae_acc = [run(ae_data) for i in range(5)]
raw_acc = [run(raw_data) for i in range(5)]

print(f"Accuracies -> AE:{np.array(ae_acc).mean()}, RAW:{np.array(raw_acc).mean()}")