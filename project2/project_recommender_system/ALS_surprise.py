from surprise import BaselineOnly
from surprise import evaluate


def als_surprise(data_train, n_epoch, reg_u, reg_i):

    print('ALS Surprise')

    bsl_options = {'method': 'als',
                   'n_epochs': n_epoch,
                   'reg_u': reg_u,
                   'reg_i': reg_i
    }

    algo = BaselineOnly(bsl_options=bsl_options)
    # Retrieve the trainset.
    trainset = data_train.build_full_trainset()

    # Build an algorithm, and train it.
    algo.train(trainset)

    perf_train = evaluate(algo, data_train, measures=['RMSE'])

    return perf_train, algo