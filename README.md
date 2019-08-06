# Movie-Recommendation

### Libraries

 * **pytorch** - for training autoencoder
 * **surprise** - recommendation library
 * **numpy** - for numerical compuations
 * **scipy** - python scientific lib
 * **tqdm** - progress bar
 * **pandas** - data handling

### Usage

``python main.py -fit``  
System runs with autoencoder and cold-start-item handler enabled. Takes some time to train autoencoder and to handle ccs items. It saves constructed dataset to *'./export_data.csv'* . After that, it continues to do what below command does. Therefore, please take a look at below.


``python main.py``  
It reads *'./export_data.csv'* file and performs recommendation evaluation with SVD, SVD++ and KNN. Each algorithm runs 5 times and then RMSE values are reported.


### Pseudocode
````
1. Create MovieLens Dataset  
2. Split CCS and NCS items (n_ratings < 50)  
3. train AE with movie contents  
4. m = pick a CCS item  
   loop until m is not CCS anymore:  
        u = pick random user  
        cos_sims = cosine similarities between latent(u.rated_movies) vs latent(m).  
       (u,m).rating = mean(cos_sims.top10())    -> fill movielens dataset accordingly.  
5.  Repeat step 4 until 0 CCS item left.  
6. Send user x movies utility matrix to surpriselib SVD++  

````

### Preliminary Results
![Proposed solution vs raw utility matrix solution with the comparison of several collaborative filtering
approaches. The experiment repeated 5 times and the result for each run is reported.](results/result.png)