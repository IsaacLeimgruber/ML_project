import scipy.sparse as sp
import numpy as np
from my_helpers import create_submission
from my_helpers import parse_row
from my_helpers import build_index_groups
# All this file code is taken from lab10
# only change is in the ALS function. We changed the parameter list, giving hyper parameters
# as parameters for cross-validation

def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - user_info.T.dot(item_info)) ** 2
    return np.sqrt(1.0 * mse / len(nz))

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""

    num_item, num_user = train.get_shape()
    print("num item", num_item)
    print("num user", num_user)
    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)

    # start by item features.
    item_nnz = train.getnnz(axis=1)
    item_sum = train.sum(axis=1)

    for ind in range(num_item):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    return user_features, item_features

def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    num_user = nnz_items_per_user.shape[0]
    num_feature = item_features.shape[0]
    lambda_I = lambda_user * sp.eye(num_feature)
    updated_user_features = np.zeros((num_feature, num_user))

    for user, items in nz_user_itemindices:
        # extract the columns corresponding to the prediction for given item
        M = item_features[:, items]

        # update column row of user features
        V = M @ train[items, user]
        A = M @ M.T + nnz_items_per_user[user] * lambda_I
        X = np.linalg.solve(A, V)
        updated_user_features[:, user] = np.copy(X.T)
    return updated_user_features


def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    num_item = nnz_users_per_item.shape[0]
    num_feature = user_features.shape[0]
    lambda_I = lambda_item * sp.eye(num_feature)
    updated_item_features = np.zeros((num_feature, num_item))

    for item, users in nz_item_userindices:
        # extract the columns corresponding to the prediction for given user
        M = user_features[:, users]
        V = M @ train[item, users].T
        A = M @ M.T + nnz_users_per_item[item] * lambda_I
        X = np.linalg.solve(A, V)
        updated_item_features[:, item] = np.copy(X.T)
    return updated_item_features


def run_ALS(train, test, num_features, lambda_user, lambda_item, vali):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    #num_features = 20  # K in the lecture notes
    #lambda_user = 0.1
    #lambda_item = 0.7
    stop_criterion = 1e-3
    change = 1
    error_list = [0, 0]

    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)

    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=0), train.getnnz(axis=1)

    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)
    # run ALS
    print("\nstart the ALS algorithm with num_features: {}, lambda_user: {}, lambda_item: {}"
          .format(num_features, lambda_user , lambda_item))
    while change > stop_criterion:
        # update user feature & item feature
        user_features = update_user_feature(
            train, item_features, lambda_user,
            nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(
            train, user_features, lambda_item,
            nnz_users_per_item, nz_item_userindices)

        error = compute_error(train, user_features, item_features, nz_train)
        error_list.append(error)
        change = np.fabs(error_list[-1] - error_list[-2])

    if(vali):
        # evaluate the test error
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
        rmse = compute_error(test, user_features, item_features, nnz_test)
        return error_list[-1], rmse, user_features, item_features
    else:
        return error_list[-1], 0, user_features, item_features

def create_ALS_pred(full_data, sub_data):
    num_feature = 20
    user_lambda = 1.0
    item_lambda = 0.01
    train_error, rmse, users, items = run_ALS(full_data, 0, num_feature, user_lambda, item_lambda, 0)
    prediction = []
    for iRow in sub_data:
        user, movie, rating = parse_row(iRow)
        item_info = items[:, movie]
        user_info = users[:, user]
        pred = user_info.T.dot(item_info)
        prediction.append(np.round([user + 1, movie + 1, pred]))
    create_submission(prediction, "prediction.csv")

def cross_val_fix_num(num_feature, train, test):
    user_lambdas = np.logspace(-2, 0, 4)
    item_lambdas = np.logspace(-2, 0, 4)
    cross_val_rmse = []
    i = 0
    for user_l in user_lambdas:
        for item_l in item_lambdas:
            train_rmse, rmse, users, items = run_ALS(train, test, num_feature, user_l, item_l, 1)
            cross_val_rmse.append([num_feature, user_l, item_l, train_rmse, rmse])
            print("iteration {} rmse:{} ".format(i, cross_val_rmse))
            i = i + 1
    np.savetxt("cross_val_fixed_f2.csv", cross_val_rmse, delimiter=",")