import pickle
import time

import numpy as np


def save_numpy(data, folderpath="/interim/", filename="results"):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    rootpath = "../../data"
    targetdir = rootpath + folderpath
    np.save(targetdir + filename + "_" + timestr + ".npy", data)

def save_pickle(data, folderpath="/interim/", filename="results"):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    rootpath = "../../data"
    targetdir = rootpath + folderpath
    with open(targetdir + filename + "_" + timestr + ".pickle", "wb") as outfile:
        pickle.dump(data, outfile)

def load_pickle(folderpath="/interim/", filename="results"):
    rootpath = "../../data"
    targetdir = rootpath + folderpath
    with open(targetdir + filename + ".pickle", "rb") as infile:
        data = pickle.load(infile)

    return data
