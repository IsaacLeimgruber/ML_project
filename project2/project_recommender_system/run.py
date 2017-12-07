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
    data_matrix = csr.csr_matrix((ratings, (rows, cols)), shape=(10000, 1000))
    print(data_matrix.nonzero())
    print(data_matrix[0,9])

    ratings = load_data(DATA_PATH);

    num_items_per_user, num_users_per_item = plot_raw_data(ratings)

    print("min # of items per user = {}, min # of users per item = {}.".format(
        min(num_items_per_user), min(num_users_per_item)))

    valid_ratings, train, test = split_data(ratings, num_items_per_user, num_users_per_item, min_num_ratings=10, p_test=0.1)





# row is in the form ["cidxcol_ridxrow", 'rating']. In order to store our data
# in a sparse matrix, we iterate over the values in data_train and parse each row.
# parse_row will return a 3-tuple (row, col, rating) where all 3 values are ints
def parse_row(row):
    row_col_str = row[0]
    rating = int(row[1])
    _idx = row_col_str.find('_')
    row = int(row_col_str[1:_idx]) - 1
    col = int(row_col_str[_idx + 2:]) - 1
    return row, col, rating


def split_data(ratings, num_items_per_user, num_users_per_item,min_num_ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
        min_num_ratings:
            all users and items we keep must have at least min_num_ratings per user and per item.
    """
    # set seed
    np.random.seed(988)

    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][:, valid_users]

    # init
    num_rows, num_cols = valid_ratings.shape
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))

    print("the shape of original ratings. (# of row, # of col): {}".format(
        ratings.shape))
    print("the shape of valid ratings. (# of row, # of col): {}".format(
        (num_rows, num_cols)))

    nz_items, nz_users = valid_ratings.nonzero()

    # split the data
    for user in set(nz_users):
        # randomly select a subset of ratings
        row, col = valid_ratings[:, user].nonzero()
        selects = np.random.choice(row, size=int(len(row) * p_test))
        residual = list(set(row) - set(selects))

        # add to train set
        train[residual, user] = valid_ratings[residual, user]

        # add to test set
        test[selects, user] = valid_ratings[selects, user]

    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test

if __name__ == '__main__':
    main()
