import numpy as np
import pandas as pd
from my_helpers import *
import xlsxwriter

DATA_PATH = "data_train.csv"

def main():
    data = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1, dtype=str)
    userId, movieId, rating = construct_data(data);

    for i in [10,20,50]:

        # we construct keep only the users and movies that have bigger than 10, 20 or 50 ratings
        # Very long to do so better to save them and load them after so we don't need to compute this everytime
        userIdX, movieIdX, ratingX = user_movie_bigger_x_ratings(userId, movieId, rating, i)

        #store them for a dataframe
        user_movie_temp = {'userId': userIdX,
                    'movieId': movieIdX,
                    'rating': ratingX}

        result_user_movie = pd.DataFrame(user_movie_temp)

        # write the dataframe on xlsx so we can keep the value
        writer = pd.ExcelWriter('keep_users_bigger_'+str(i)+'_rating.xlsx', engine='xlsxwriter')
        result_user_movie.to_excel(writer, 'Sheet1')
        writer.save()

if __name__ == '__main__':
    main()
