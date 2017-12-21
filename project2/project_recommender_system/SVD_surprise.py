from surprise import evaluate
from surprise import SVD


def svd_surprise(data_train, reg_all, init_mean, n_epochs, lr_all, n_factors):

    print('SVD Surprise')

    algo = SVD(reg_all= reg_all, init_mean = init_mean, n_epochs = n_epochs, lr_all= lr_all, n_factors = n_factors)

    # Retrieve the trainset.
    trainset = data_train.build_full_trainset()

    # Build an algorithm, and train it.
    algo.train(trainset)

    perf_train = evaluate(algo, data_train, measures=['RMSE'])

    return perf_train, algo