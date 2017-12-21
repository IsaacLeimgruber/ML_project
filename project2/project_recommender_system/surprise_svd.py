from surprise import evaluate
from surprise import SVD
from my_helpers import make_prediction_surprise


def svd_surprise(data_train, reg_all, init_mean, n_epochs, lr_all, n_factors, name_file):
    print('SVD Surprise')

    # We construct our SVD algo with surprise and the best parameters
    algo = SVD(reg_all= reg_all, init_mean = init_mean, n_epochs = n_epochs, lr_all= lr_all, n_factors = n_factors)

    # Retrieve the trainset.
    trainset = data_train.build_full_trainset()

    # Build an algorithm, and train it.
    algo.train(trainset)
    # Evaluate the RMSE of the algo
    evaluate(algo, data_train, measures=['RMSE'])
    # Make the prediction
    make_prediction_surprise(algo, name_file)