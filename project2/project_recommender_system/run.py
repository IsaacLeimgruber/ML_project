import numpy as np
import scipy.sparse as csr
from random import shuffle
from ALS import run_ALS
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

    userId, movieId, rating = construct_data(data);

    indices_to_shuffle = np.array(range(len(userId)))
    print("test")

    X_train_userId, X_train_movieId, X_train_rating, X_test_userId, X_test_movieId, X_test_rating = split_data(indices_to_shuffle, userId, movieId, rating, 70)

    data_XTrain = csr.csr_matrix((X_train_rating, (X_train_userId, X_train_movieId)), shape=(10000, 1000))
    data_XTest = csr.csr_matrix((X_test_rating, (X_test_userId, X_test_movieId)), shape=(10000, 1000))
    #print((data_XTrain.T).index(928))
    # attention no movie 928?

    print("TEST ALS")

    rmse, users, items = run_ALS(data_XTrain, data_XTest, 20, 0.1, 0.7)
    print("rmse als", rmse)
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

    #create_submission(prediction, "GLOBAL_MOVIE_AVG.csv")
    #create_submission(prediction, "GLOBAL_AVG.csv")

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