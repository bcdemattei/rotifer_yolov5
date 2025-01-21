import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Time1553_distances.csv")


img = plt.hist(df)
plt.imsave("test.png", img)