from surprise import BaselineOnly
from surprise import evaluate
from my_helpers import make_prediction_surprise


def knn_surprise(data_train, n_epoch, reg_u, reg_i, name_file):
    print('ALS Surprise')

    #We construct our KNN algo with surprise and the best parameters
    bsl_options = {'method': 'als',
                   'n_epochs': n_epoch,
                   'reg_u': reg_u,
                   'reg_i': reg_i
    }

    #Create algo KNN BaselineOnly
    algo = BaselineOnly(bsl_options=bsl_options)
    # Retrieve the trainset.
    trainset = data_train.build_full_trainset()

    #Build an algorithm, and train it.
    algo.train(trainset)

    #Evaluate the RMSE of the algo
    evaluate(algo, data_train, measures=['RMSE'])
    # Make the prediction
    make_prediction_surprise(algo, name_file)