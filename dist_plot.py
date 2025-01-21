import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib as Path

def process_and_plot_distances(file_path, source):

    p = Path(source)

    df = pd.read_csv(file_path)
    df.drop(df.columns[[0]], axis = 1, inplace=True)
    df = df.to_numpy()
    df = df.transpose()


    img = plt.hist(df[~np.isnan(df)], bins = 20)
    plt.xlabel("Normalized Distance of C. erosa to Bdelloid")
    plt.ylabel("Frequency of Distances Across All Frames")
    plt.title(f"{p.stem} Distances")
    plt.show(img)

    print(f"Max normalized distance: {np.nanmax(df)}")
    print(f"Min normalized distance: {np.nanmin(df)}")