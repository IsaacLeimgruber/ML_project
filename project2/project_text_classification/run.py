import numpy as np
import matplotlib.pyplot as plt
DATA_FOLDER = "Data/"
# https://www.overleaf.com/12522751fsgrdpydrwxf READ and EDIT link for report
# tes merge


def main():
    print("hello")
    # data_path = DATA_FOLDER + "data.csv"
    data_path = "sample_submission.csv"
    data = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    print(data)


if __name__ == '__main__':
    main()
