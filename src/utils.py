import pickle
import time
from typing import Any

import numpy as np
import numpy.typing as npt


def save_numpy(
        data: npt.NDArray[np.float32],
        folderpath: str = "/interim/",
        filename: str = "results"
) -> None:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    rootpath = "../../data"
    targetdir = rootpath + folderpath
    np.save(targetdir + filename + "_" + timestr + ".npy", data)

def save_pickle(
        data: Any,
        folderpath: str = "/interim/",
        filename: str = "results"
) -> None:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    rootpath = "../../data"
    targetdir = rootpath + folderpath
    with open(targetdir + filename + "_" + timestr + ".pickle", "wb") as outfile:
        pickle.dump(data, outfile)

def load_pickle(
        folderpath: str = "/interim/",
        filename: str = "results"
) -> Any:
    rootpath = "../../data"
    targetdir = rootpath + folderpath
    with open(targetdir + filename + ".pickle", "rb") as infile:
        data = pickle.load(infile)

    return data