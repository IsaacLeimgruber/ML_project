import numpy as np
import scipy.sparse as csr
from random import shuffle
from plots import plot_raw_data
from helpers import load_data, preprocess_data
import scipy.sparse as sp
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# https://www.overleaf.com/12522751fsgrdpydrwxf READ and EDIT link for report
DATA_PATH = "data_train.csv"
DATA_PATH_SUB = "sample_submission.csv"


def main():
    data = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1, dtype=str)
    sub_data = np.genfromtxt(DATA_PATH_SUB, delimiter=",", skip_header=1, dtype=str)
    print(data)
    print(len(data))
    print("c10_r20"[1:3])
    print(parse_row(["c10_r20", '4']))

    userId, movieId, rating = construct_data(data);

    indices_to_shuffle = range(len(userId))
    print("test")

    X_train_userId, X_train_movieId, X_train_rating, X_test_userId, X_test_movieId, X_test_rating = split_data(indices_to_shuffle, userId, movieId, rating, 100)

    data_XTrain = csr.csr_matrix((X_train_rating, (X_train_userId, X_train_movieId)), shape=(10000, 1000))
    data_XTest = csr.csr_matrix((X_test_rating, (X_test_userId, X_test_movieId)), shape=(10000, 1000))
    print(data_XTrain.nonzero())
    print(data_XTrain[0,9])
    print(data_XTrain.nonzero()[0])
    print(data_XTrain.nonzero()[1])
    #print(data_matrix_movie.index(928))
    # attention no movie 928?

    avgUser, avgMovie, avgGlobal = calculate_averages(data_XTrain)

    print(avgUser)
    print(avgMovie)
    print(avgGlobal)

    prediction = []
    for iRow in sub_data:
        user, movie, rating = parse_row(iRow)
        rate = int(np.round(avgMovie[movie]))
        #rate = avgGlobal
        prediction.append([user+1, movie+1, rate])

    create_submission(prediction, "GLOBAL_MOVIE_AVG.csv")
    #create_submission(prediction, "GLOBAL_AVG.csv")

    print("TEST SGD")

    sgd(data_XTrain, data_XTest)


def init_MF(train, num_features):
    """init the parameter for matrix factorization."""

    num_item, num_user = train.get_shape()

    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)

    # start by item features.
    item_nnz = train.getnnz(axis=1)
    item_sum = train.sum(axis=1)

    for ind in range(num_item):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    return user_features, item_features


def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - user_info.T.dot(item_info)) ** 2
    return np.sqrt(1.0 * mse / len(nz))


def sgd(train, test):
    gamma = 0.01
    num_features = 20  # K in the lecture notes
    lambda_user = 0.1
    lambda_item = 0.7
    num_epochs = 20  # number of full passes through the train set
    errors = [0]

    user_features, item_features = init_MF(train, num_features)

    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))


    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.2

        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            err = train[d, n] - user_info.T.dot(item_info)

            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))

        errors.append(rmse)

    # evaluate the test error
    rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(rmse))




def split_data(idx, userId, movieId, rating, perc_train):
    shuffle(idx)
    idx_te = int(perc_train * len(idx) / 100.0)
    X_train_idx = idx[0:idx_te]
    X_test_idx = idx[idx_te:len(idx)]

    X_train_userId = [userId[i] for i in X_train_idx]
    X_train_movieId = [movieId[i] for i in X_train_idx]
    X_train_rating = [rating[i] for i in X_train_idx]
    X_test_userId = [userId[i] for i in X_test_idx]
    X_test_movieId = [movieId[i] for i in X_test_idx]
    X_test_rating = [rating[i] for i in X_test_idx]

    return X_train_userId, X_train_movieId, X_train_rating, X_test_userId, X_test_movieId, X_test_rating

def construct_data(data_):
    rows = []
    cols = []
    ratings = []
    for iRow in data_:
        row, col, rating = parse_row(iRow)
        rows.append(row)
        cols.append(col)
        ratings.append(rating)
    return rows,cols,ratings

def calculate_averages(data_user):
    one_user = np.ones(np.shape(data_user)[0])
    one_movie = np.ones(np.shape(data_user)[1])

    sum_movie = (data_user.T).dot(one_user)
    sum_user = data_user.dot(one_movie)

    columns_user = (data_user != 0).sum(1)
    columns_movie = ((data_user != 0).sum(0)).T

    avgUser = {}
    resultUser = np.zeros(np.shape(sum_user))
    for i in range(len(columns_user)):
        avgUser[i] = sum_user.T[i] / columns_user[i,0]
        resultUser[i] = sum_user.T[i] / columns_user[i,0]


    avgMovie = {}
    resultMovie = np.zeros(np.shape(sum_movie))
    for i in range(len(columns_movie)):
        if(columns_movie[i, 0] != 0):
            avgMovie[i] = sum_movie.T[i] / columns_movie[i, 0]
            resultMovie[i] = sum_movie.T[i] / columns_movie[i, 0]
        else:
            avgMovie[i] = sum_movie.T[i]
            resultMovie[i] = sum_movie.T[i]

    avgGlobal_user = np.sum(resultUser) / np.shape(resultUser)[0]
    avgGlobal_movie = np.sum(resultMovie) / np.shape(resultMovie)[0]

    avgGlobal = (avgGlobal_user + avgGlobal_movie) / 2
    #avgGlobal = int(np.round(avgGlobal_movie))

    return avgUser, avgMovie, avgGlobal


def parse_row(row):
    row_col_str = row[0]
    rating = int(row[1])
    _idx = row_col_str.find('_')
    row = int(row_col_str[1:_idx]) - 1
    col = int(row_col_str[_idx + 2:]) - 1
    return row, col, rating

def create_submission(data, filename="submission.csv"):

    print("Creating submission " + str(filename))
    f = open(filename,"w")
    f.write("Id,Prediction\n")
    for user, movie, rating in data:
        f.write('r{0}_c{1},{2}'.format(user,movie,rating) + "\n")
    f.close()


if __name__ == '__main__':
    main()
