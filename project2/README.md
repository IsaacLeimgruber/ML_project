## How tu run our code
You can obtain our result on kaggle by running the program run.py 
which will execute ALS on the data with the best hyperparameters we found
with the cross-validation. The predictions will be in the file prediction.csv

## KNN version
You can run the KNN version by calling KNN_surprise which will create a prediction in the file which filename is given as parameter

## SVD version
Call the SVD_surprise function to run the SVD algorithm and create a submission in the same fashion as the KNN version

### Python files
You can find below a brief description of the pyhton files you can find on the repository

#### my_helpers.py
Contains multiple helper function
* parse_row: extracts row, column and rating values from a formatted data row for example from "r10_c20, 3" extracts 10, 20, 3
* build_index_groups: creates ratings, rows, cols from train matrix
* group_by: groups bi_dimensional lists by a specific index
* create_submission: creates a submission from an array of 3-tuple in the form (user, movie, rating)
* rmse: computes the rmse between a and b
* compute_baselines: computes the rmse for the different averages (user, movie, global) with respect to data (usually, compute average
on data train then compute baselines with data test
* split_data: splits the users, movies and ratings randomly with a ratio of perc_train. This function should be called with idx = range(len(nnz[0]), nnz[0], nnz[1], data[nnz], ratio where nnz is data.nonzero() and data is a csr matrix
* construct_data: extracts the rows, cols and ratings from an array data. construct_data applies parse_row iteratively
* calculate_averages: computes the user and movie averages of the data, returns two array of length K and D where shape(data) = [K, D]
* calculate_mse: computes the mse between the labels and the predictions

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
