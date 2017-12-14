import numpy as np
import scipy.sparse as csr
from itertools import groupby

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
