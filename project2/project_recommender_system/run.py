import numpy as np
import scipy.sparse as csr
from plots import plot_raw_data
from helpers import load_data, preprocess_data
import scipy.sparse as sp
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# https://www.overleaf.com/12522751fsgrdpydrwxf READ and EDIT link for report
DATA_PATH = "data_train.csv"


def main():
    data = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1, dtype=str)
    print(data)
    print(len(data))
    print("c10_r20"[1:3])
    print(parse_row(["c10_r20", '4']))
    rows = []
    cols = []
    ratings = []
    for iRow in data:
        row, col, rating = parse_row(iRow)
        rows.append(row)
        cols.append(col)
        ratings.append(rating)
    data_matrix_user = csr.csr_matrix((ratings, (rows, cols)), shape=(10000, 1000))
    data_matrix_movie = csr.csr_matrix((ratings, (cols, rows)), shape=(1000, 10000))
    print(data_matrix_user.nonzero())
    print(data_matrix_movie.nonzero())
    print(data_matrix_user[0,9])
    print(data_matrix_user.nonzero()[0])
    print(data_matrix_user.nonzero()[1])

    avgUser, avgMovie, avgGlobal = calculate_averages(data_matrix_user, data_matrix_movie)

    print(avgUser)
    print(avgMovie)
    print(avgGlobal)

    prediction = np.zeros((10000,1000))

    for i in range(len(prediction)):
        if(i % 1000 == 0):
            print(i)
        for j in range(len(prediction[i])):
            if(data_matrix_user[i,j] != 0):
                prediction[i][j] = data_matrix_user[i,j]
            else:
                prediction[i][j] = avgGlobal

    print(prediction)

    create_submission(prediction, "GLOBAL_AVG.csv")



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
        avgMovie[i] = sum_movie.T[i] / columns_movie[i, 0]
        resultMovie[i] = sum_movie.T[i] / columns_movie[i, 0]

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
    avgGlobal = (avgGlobal_user + avgGlobal_movie) / 2

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
    for userIdx in range(len(data)):
        for movieIdx in range(len(data[userIdx])):
            rating = data[userIdx][movieIdx]
            f.write('r{0}_c{1},{2}'.format(userIdx,movieIdx,rating) + "\n")
    f.close()


if __name__ == '__main__':
    main()
