import numpy as np
import scipy.sparse as csr
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


if __name__ == '__main__':
    main()
