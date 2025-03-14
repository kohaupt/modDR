import numpy as np
import pandas as pd

from dash_overlay import DashOverlay

if __name__ == "__main__":
    np_array = np.load("../data/interim/results_20250314-131448.npy", allow_pickle=True)
    df_list = []
    for i in range(len(np_array)):
        df_list.append(pd.DataFrame(np_array[i], columns=["x", "y", "metrics_score"]))

    app = DashOverlay(df_list, [0, 1, 3, 5])
    app.run(debug=True)
