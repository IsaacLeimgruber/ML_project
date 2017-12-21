## How tu run our code
You can obtain our result on kaggle by running the program run.py 
which will execute ALS on the data with the best hyperparameters we found
with the cross-validation. The predictions will be in the file prediction.csv

## KNN version
You can run the KNN version by calling knn_surprise from surprise_knn which will create a prediction in the file which filename is given as parameter

## SVD version
Call the svd_surprise function from surprise_svd to run the SVD algorithm and create a submission in the same fashion as the KNN version

### Python files
You can find below a brief description of the pyhton files you can find on the repository

#### my_helpers.py
Contains multiple helper function
* construct_data: extracts the rows, cols and ratings from an array data. construct_data applies parse_row iteratively
* parse_row: extracts row, column and rating values from a formatted data row for example from "r10_c20, 3" extracts 10, 20, 3
* build_index_groups: creates ratings, rows, cols from train matrix
* group_by: groups bi_dimensional lists by a specific index
* create_submission: creates a submission from an array of 3-tuple in the form (user, movie, rating)
* rmse: computes the rmse between a and b
* compute_baselines: computes the rmse for the different averages (user, movie, global) with respect to data (usually, compute average
on data train then compute baselines with data test
* split_data: splits the users, movies and ratings randomly with a ratio of perc_train. This function should be called with idx = range(len(nnz[0]), nnz[0], nnz[1], data[nnz], ratio where nnz is data.nonzero() and data is a csr matrix
* calculate_averages: computes the user and movie averages of the data, returns two array of length K and D where shape(data) = [K, D]
* calculate_mse: computes the mse between the labels and the predictions
* make_prediction_surprise: make a prediction when we call knn_surprise from surprise_knn or svd_surprise from surprise_svd
* user_movie_bigger_x_ratings: This function will remove all the elements that appears less than x times. For us, it will remove all user and all movies that appears less than x times. It looks at the indices and delet them if the sum is less than x. it was very important to mix the indices otherwise the ratings would no longer belong to the same user or movie...

#### ALS.py
Contains ALS implementation (not the surprise one)
* compute_error: computes the rmse for given parameters
* init_MF: initialize the parameters for the matrix factorization
* update_user_feature: makes one step for the user features
* update_item_feature: makes one step for the item features
* run_ALS: consecutively update user and item features until the reduction of error between two iterations becomes smaller than 
the change threshold
* create_ALS_pred: creates a prediction using ALS and the cross-validated hyperparameters. The prediction is written in prediction.csv
takes between 10 and 20 minutes to run
* cross_val_fix_num: cross-validates using ALS and train and validation data. The lambdas are set as np.logspace(-2, 0, 4). You can 
change the lambdas number of values or range as your patience increases

### bigger_than_x_rating.py
Contains user_movie_bigger_x_ratings function.
It will create files xlsx with the users and movies that have more than 10, 20 or 50 ratings
It's very long to compute, so we save the results and let them (under the name "users_movies_bigger_X_rating.xlsx") in the folder
We can decide to chooses other parameters easily and it will create an excel file with a correct name
We implemented this because we wanted to see if we remove the movies and the users who have few evaluations, would the rmse decrease?
We can load the result easily and don't need to compute this everytime (very long).

### surprise_knn.py
Contains knn_surprise
It is everything form the library surprise to compute an KNN algorithm, train on the data and make a prediction ready for Kaggle.

### surprise_svd.py
Contains svd_surprise
It is everything form the library surprise to compute a SVD algorithm, train on the data and make a prediction ready for Kaggle.
We implemented this algorithm because it's one of the best of the surprise library and we thought it would give us better results.

### surprise_cross_validation.py
Contains functions that make grid search on a hyper parameter list to determine the best hyper parameter (on surprise).
* grid_search_knn_surprise: On a list of hyper parameters, it will create the Grid search algo (from surprise) and return the best hyper parameters (for the function knn_surprise)
* grid_search_knn: On a list of parameters, we do manually a Cross validation on KNN algo from surprise (and stores the results of training and testing sets). We do it manually because we wanted to see all the results and not only the best one as the GridSearch function of surprise does. This is to study each case and perhaps to make a graphs of the results. It stores every RMSE mean and all hyper parameters in a xslx file.
* grid_search_svd_surprise: On a list of hyper parameters, it will create the Grid search algo (from surprise) and return the best hyper parameters (for the function svd_surprise)
* grid_search_svd: On a list of parameters, we do manually a Cross validation on SVD algo from surprise (and stores the results of training and testing sets).It stores every RMSE mean and all hyper parameters in a xslx file and we do it for the same reason as grid_search_knn (store and see every results)

### surprise_run.py
This run contains everything to Grid search the hyper parameters for KNN and SVD surprise and make prediction.
If you want to see the results you just have to run this and it will do the GridSearch from surprise (and the one we created too) After that it will run knn_surprise and svd_surprise to make the best prediction based on the best hyper parameters found with the GridSearch from surprise library.
First it will load the data from the file, create a X_train and X_test form the users, movies and ratings (train and test)
Subsequently we will choose our hyper parameters for the knn grid_search and do the grid search from surprise with the function grid_search_knn_surprise.
After that we will compute the same grid search (grid_search_knn) and save all the parameters and finally call the knn_surprise and make a prediction.
The same thing will appears for svd algorithm.
But that's not all. We can see that this loop will be repeated a certain number of times to apply the same algorithms on different data. Indeed, the first time, all this will be applied to the normal data (without modifications). Other times, we are going to use our saved results with the user_movie_bigger_x_ratings function. This process is therefore repeated for data where users and movies that have less than 10, 20 or 50 ratings have been deleted. So we can see if the result will be improve and all the submission files will be done.
