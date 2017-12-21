import numpy as np
import scipy.sparse as csr
from itertools import groupby
from random import seed
from random import shuffle
from itertools import groupby

def parse_row(row):
    row_col_str = row[0]
    rating = float(row[1])
    space_idx = row_col_str.find('_')
    row = int(row_col_str[1:space_idx]) - 1
    col = int(row_col_str[space_idx + 2:]) - 1
    return row, col, rating


# def make_matrix(data):
#     rows = []
#     cols = []
#     ratings = []
#     max_row = 0
#     max_col = 0
#     for iRow in data:
#         row, col, rating = parse_row(iRow)
#         rows.append(row)
#         cols.append(col)
#         ratings.append(rating)
#         max_row = max(row, max_row)
#         max_col = max(col, max_col)
#     print(max_row, max_col)
#     return csr.csr_matrix((ratings, (rows, cols)), shape=[max_row + 1, max_col + 1])


# Taken from helpers from lab10
def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def create_submission(data, filename="submission.csv"):
    print("Creating submission " + str(filename))
    f = open(filename, "w")
    f.write("Id,Prediction\n")
    for user, movie, rating in data:
        f.write('r{0}_c{1},{2}'.format(int(user), int(movie), int(rating)) + "\n")
    f.close()

def rmse(a, b):
    return np.sqrt(2 * (a - b) ** 2)


def compute_baselines(user_avg, movie_avg, global_avg, data):
    user_rmse = 0
    movie_rmse = 0
    global_rmse = 0
    nnz = data.nonzero()
    for idx in range(len(nnz[0])):
            user = nnz[1][idx]
            movie = nnz[0][idx]
            val = data[movie, user]
            user_rmse = user_rmse + rmse(user_avg[user], val)
            movie_rmse = movie_rmse + rmse(movie_avg[movie], val)
            global_rmse = global_rmse + rmse(global_avg, val)
    user_rmse = user_rmse/len(nnz[0])
    movie_rmse = movie_rmse/len(nnz[0])
    global_rmse = global_rmse/len(nnz[0])
    return user_rmse, movie_rmse, global_rmse

def split_data(idx, userId, movieId, rating, perc_train):
    seed(42)
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
    return avgUser, avgMovie

def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)