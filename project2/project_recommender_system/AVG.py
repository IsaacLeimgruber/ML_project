import numpy as np

def avg(data_user):
    one_user = np.ones(np.shape(data_user)[0])
    one_movie = np.ones(np.shape(data_user)[1])

    sum_movie = (data_user.T).dot(one_user)
    sum_user = data_user.dot(one_movie)

    columns_user = (data_user != 0).sum(1)
    columns_movie = ((data_user != 0).sum(0)).T

    avgUser = {}
    resultUser = np.zeros(np.shape(sum_user))
    for i in range(len(columns_user)):
        if (columns_user[i, 0] != 0):
            avgUser[i] = sum_user.T[i] / columns_user[i,0]
            resultUser[i] = sum_user.T[i] / columns_user[i,0]
        else:
            avgUser[i] = sum_user.T[i]
            resultUser[i] = sum_user.T[i]


    avgMovie = {}
    resultMovie = np.zeros(np.shape(sum_movie))
    for i in range(len(columns_movie)):
        if(columns_movie[i, 0] != 0):
            avgMovie[i] = sum_movie.T[i] / columns_movie[i, 0]
            resultMovie[i] = sum_movie.T[i] / columns_movie[i, 0]
        else:
            avgMovie[i] = sum_movie.T[i]
            resultMovie[i] = sum_movie.T[i]

    avgGlobal_user = np.sum(resultUser) / np.shape(resultUser)[0]
    avgGlobal_movie = np.sum(resultMovie) / np.shape(resultMovie)[0]

    avgGlobal = (avgGlobal_user + avgGlobal_movie) / 2
    #avgGlobal = int(np.round(avgGlobal_movie))

    return avgUser, avgMovie, avgGlobal
