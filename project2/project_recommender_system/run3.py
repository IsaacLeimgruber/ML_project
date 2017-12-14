import numpy as np
import scipy.sparse as csr
from my_helpers import make_matrix
from sklearn.model_selection import train_test_split

# https://www.overleaf.com/12522751fsgrdpydrwxf READ and EDIT link for report

DATA_PATH_TRAIN = "data_train.csv"
DATA_PATH_SUB = "sample_submission.csv"


def main():
    data = np.genfromtxt(DATA_PATH_TRAIN, delimiter=",", skip_header=1, dtype=str)
    ratings = make_matrix(data)
    print("ratings", ratings)
    nnz = ratings.nonzero()
    print("nnz",nnz)
    global_mean = ratings[nnz].mean()
    print("glob_mean", global_mean)
    print("ratings[nnz]", ratings[nnz])
    print("nnz[0]", list(set(nnz[0])))
    print("ratings[nnz[0]]", ratings.getnnz(0)[1])
    #nnz_users = ratings[nnz[0]]
    #print(nnz_users)
    user_means, movie_means = compute_user_movie_avg(nnz, ratings)
    print("user_means", user_means)
    print("movie_means", movie_means)

def compute_user_movie_avg(nnz, ratings):
    users, movies = nnz
    user_means = np.zeros(ratings.get_shape()[0])
    movie_means = np.zeros(ratings.get_shape()[1])
    entries = zip(users, movies)
    print(entries)
    for entry in entries:
        rating = ratings[entry[0], entry[1]]
        user_means[entry[0]] = user_means[entry[0]] + rating
        movie_means[entry[1]] = movie_means[entry[1]] + rating
    for user in list(set(users)):
        user_means[user] = user_means[user]/ratings.getnnz(axis=1)[user]
    for movie in list(set(movies)):
        movie_means[movie] = movie_means[movie]/ratings.getnnz(axis=0)[movie]
    return user_means, movie_means

if __name__ == '__main__':
    main()
