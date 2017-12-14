import numpy as np
import scipy.sparse as csr
from itertools import groupby
from random import shuffle

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

def parse_row(row):
    row_col_str = row[0]
    rating = float(row[1])
    space_idx = row_col_str.find('_')
    row = int(row_col_str[1:space_idx]) - 1
    col = int(row_col_str[space_idx + 2:]) - 1
    return row, col, rating

def make_matrix(data):
    rows = []
    cols = []
    ratings = []
    max_row = 0
    max_col = 0
    for iRow in data:
        row, col, rating = parse_row(iRow)
        rows.append(row)
        cols.append(col)
        ratings.append(rating)
        max_row = max(row, max_row)
        max_col = max(col, max_col)
    print(max_row, max_col)
    return csr.csr_matrix((ratings, (rows, cols)), shape = [max_row + 1, max_col + 1])

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
    f = open(filename,"w")
    f.write("Id,Prediction\n")
    for user, movie, rating in data:
        f.write('r{0}_c{1},{2}'.format(user,movie,rating) + "\n")
    f.close()
