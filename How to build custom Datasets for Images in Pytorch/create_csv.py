import pandas as pd
import os

root = "../dataset/CATDOG/"
filenames = os.listdir(root + "train")
labels = [0]*25000

for i in range(len(filenames)):
    if filenames[i][0] == "d":
        labels[i] = 1

df = pd.DataFrame([filenames, labels]).T
df.to_csv(root + "labels.csv", index=False)