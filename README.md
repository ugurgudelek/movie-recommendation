# movie-recommendation

## TODOs
 * IMDB dataseti gibi bir dataset kullanarak, NCS ve ICS filmlerinden feature çıkartmak lazım (without user-ratings) - Ama belki de movielens datasetinin featureları kendi başına yeterli olur.
 * Movielens datasetinden user ratingslerini atarak CCS leri kullanabiliriz. User-ratings yerine de cosine similarity ile ICS ve NCS  filmlerinden çekebiliriz.
 * Featurelarının pd.DataFrame şeklinde topladığımız verileri AutoEncoder a sokup, reconstruction error belli bir seviyenin altına inince, latent space'ini çıkarıp CF için kullanalım.
 * CF için ise şimdilik timeSVD++.


## Type of AEs
* Sparse Autoencoders: It uses regularisation by putting a penalty on the loss function. At any time an AutoEncoder can use only a limited units of the hidden layer
* Denoising Autoencoder: Randomly turn some of the units of the first hidden layers to zero. This is a stochastic AutoEncoder.
* Stacked AutoEncoders: They can superseed the results of Deep Belief Networks and are made up of multiple encoding and decoding layers.
* Deep AutoEncoders: These are Stacked Restricted Boltzman Machines.
* Convolutional autoencoder
* Sequence-to-sequence autoencoder
* Variational autoencoder (VAE): generative model


