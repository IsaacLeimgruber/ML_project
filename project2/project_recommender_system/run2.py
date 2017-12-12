import numpy as np
import scipy.sparse as csr
from random import shuffle
from plots import plot_raw_data
from helpers import load_data, preprocess_data
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# https://www.overleaf.com/12522751fsgrdpydrwxf READ and EDIT link for report
DATA_PATH = "data_train.csv"


def main():
    data = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1, dtype=str)
    print(data)
    print(len(data))
    print("c10_r20"[1:3])
    print(parse_row(["c10_r20", '4']))

    X_train_csv, X_test_csv = split_data(data, 70)
    print(len(X_train_csv) + len(X_test_csv))

    X_train_userId, X_train_movieId, X_train_rating = construct_data(X_train_csv);
    X_test_userId, X_test_movieId, X_test_rating = construct_data(X_test_csv);

    #print(X_train_userId)
    #print(X_train_movieId)
    print(X_train_rating)

    #X_train_matrix = csr.csr_matrix((X_train_rating, (X_train_userId, X_train_movieId)), shape=(10000, 1000))
    #X_test_matrix = csr.csr_matrix((X_test_rating, (X_test_userId, X_test_movieId)), shape=(10000, 1000))
    x = [[1],[2],[3]]
    print(x)
    create_submission(x, "GLOBAL_AVG.csv")



# row is in the form ["cidxcol_ridxrow", 'rating']. In order to store our data
# in a sparse matrix, we iterate over the values in data_train and parse each row.
# parse_row will return a 3-tuple (row, col, rating) where all 3 values are ints
def construct_data(data):
    rows = []
    cols = []
    ratings = []
    for iRow in data:
        row, col, rating = parse_row(iRow)
        rows.append(row)
        cols.append(col)
        ratings.append(rating)
    return rows,cols,rating

def split_data(data, perc_train):
    shuffle(data)
    idx_te = int(perc_train * len(data) / 100.0)
    X_train = data[0:idx_te]
    X_test = data[idx_te:len(data)]

    return X_train, X_test


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
