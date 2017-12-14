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

    X_train_csv, X_test_csv = split_data(data, 100)
    print(len(X_train_csv) + len(X_test_csv))

    X_train_userId, X_train_movieId, X_train_rating = construct_data(data);

    data_matrix_user = csr.csr_matrix((X_train_rating, (X_train_userId, X_train_movieId)), shape=(10000, 1000))
    data_matrix_movie = csr.csr_matrix((X_train_rating, (X_train_movieId, X_train_userId)), shape=(1000, 10000))
    print(data_matrix_user.nonzero())
    print(data_matrix_movie.nonzero())
    print(data_matrix_user[0,9])
    print(data_matrix_user.nonzero()[0])
    print(data_matrix_user.nonzero()[1])
    #print(data_matrix_movie.index(928))
    # attention no movie 928?

    avgUser, avgMovie, avgGlobal = calculate_averages(data_matrix_user, data_matrix_movie)

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


def split_data(data, perc_train):
    shuffle(data)
    idx_te = int(perc_train * len(data) / 100.0)
    print(idx_te)
    X_train = data[0:idx_te]
    X_test = data[idx_te:len(data)]

    return X_train, X_test

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

def calculate_averages(data_user, data_movie):
    user = data_user.nonzero()[0]
    movie = data_movie.nonzero()[0]

    one_user = np.ones(np.shape(data_user)[1])
    one_movie = np.ones(np.shape(data_movie)[1])

    sum_user = data_user.dot(one_user)
    sum_movie = data_movie.dot(one_movie)

    columns_user = (data_user != 0).sum(1)
    columns_movie = (data_movie != 0).sum(1)

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

    '''
    for i in range(len(user)):
        if(i % 100000 == 0):
            print(i)

        if(i != 0):
            if(user[i] == user[i-1]):
                sum_user = sum_user + data_user[user[i],data_user.nonzero()[1][i]]
            else:
                avgUser[user[i]] = sum_user / nb_rating_user
                avgGlobal_user = avgGlobal_user + avgUser[user[i]]
                sum_user = data_user[user[i],data_user.nonzero()[1][i]]
                nb_rating_user = 0

            if (movie[i] == movie[i - 1]):
                sum_movie = sum_movie + data_movie[movie[i], data_movie.nonzero()[1][i]]
            else:
                avgMovie[movie[i]] = sum_movie / nb_rating_movie
                avgGlobal_movie = avgGlobal_movie + avgMovie[movie[i]]
                sum_movie = data_movie[movie[i], data_movie.nonzero()[1][i]]
                nb_rating_movie = 0

        else:
            sum_user = data_user[user[i],data_user.nonzero()[1][i]]
            sum_movie = data_user[movie[i], data_movie.nonzero()[1][i]]
            print(sum_user)
            print(movie)

        nb_rating_user = nb_rating_user + 1
        nb_rating_movie = nb_rating_movie + 1
    '''
    #avgGlobal = (avgGlobal_user + avgGlobal_movie) / 2
    avgGlobal = int(np.round(avgGlobal_movie))

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
