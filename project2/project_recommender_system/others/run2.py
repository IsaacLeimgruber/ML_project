import numpy as np
import scipy.sparse as csr
from my_helpers import *
from SGD import sgd
from AVG import avg
#from plots import plot_raw_data
#from helpers import load_data, preprocess_data
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# https://www.overleaf.com/12522751fsgrdpydrwxf READ and EDIT link for report
DATA_PATH = "data_train.csv"
DATA_PATH_SUB = "sample_submission.csv"


def main():
    data = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1, dtype=str)
    sub_data = np.genfromtxt(DATA_PATH_SUB, delimiter=",", skip_header=1, dtype=str)
    print(data)
    print(len(data))
    print("c10_r20"[1:3])
    print(parse_row(["c10_r20", '4']))

    userId, movieId, rating = construct_data(data);

    indices_to_shuffle = np.array(range(len(userId)))
    print("test")

    test_ratio = 70
    X_train_userId, X_train_movieId, X_train_rating, X_test_userId, X_test_movieId, X_test_rating = split_data(indices_to_shuffle, userId, movieId, rating, test_ratio)

    data_XTrain = csr.csr_matrix((X_train_rating, (X_train_userId, X_train_movieId)), shape=(10000, 1000))
    data_XTest = csr.csr_matrix((X_test_rating, (X_test_userId, X_test_movieId)), shape=(10000, 1000))
    print(data_XTrain.nonzero())
    print(data_XTrain[0,9])
    print(data_XTrain.nonzero()[0])
    print(data_XTrain.nonzero()[1])
    #print(data_matrix_movie.index(928))
    # attention no movie 928?

    avgUser, avgMovie, avgGlobal = avg(data_XTrain)

    print(avgUser)
    print(avgMovie)
    print(avgGlobal)

    prediction = []
    for iRow in sub_data:
        user, movie, rating = parse_row(iRow)
        rate = int(np.round(avgMovie[movie]))
        #rate = avgGlobal
        prediction.append([user+1, movie+1, rate])

    create_submission(prediction, "GLOBAL_MOVIE_AVG.csv")
    #create_submission(prediction, "GLOBAL_AVG.csv")

    print("TEST SGD")

    #sgd(data_XTrain, data_XTest)


if __name__ == '__main__':
    main()
