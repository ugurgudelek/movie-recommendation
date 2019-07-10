# movie-recommendation

 * IMDB dataseti gibi bir dataset kullanarak, NCS ve ICS filmlerinden feature çıkartmak lazım (without user-ratings) - Ama belki de movielens datasetinin featureları kendi başına yeterli olur.
 * Movielens datasetinden user ratingslerini atarak CCS leri kullanabiliriz. User-ratings yerine de cosine similarity ile ICS ve NCS  filmlerinden çekebiliriz.
 * Featurelarının pd.DataFrame şeklinde topladığımız verileri AutoEncoder a sokup, reconstruction error belli bir seviyenin altına inince, latent space'ini çıkarıp CF için kullanalım.
 * CF için ise şimdilik timeSVD++.


