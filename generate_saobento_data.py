from utils import filter_data, polyline_to_df, DATA_DIR
import pandas as pd
import numpy as np
import time

INPUT_FILE = DATA_DIR + "train.csv"
OUTPUT_FILE = DATA_DIR + "Sao_bento.pkl"

print("Generate Sao Bento dataset from " + INPUT_FILE + " file.")

ts = time.time()
train_filter = [filter_data(pl, t) for pl, t in pd.read_csv(INPUT_FILE, sep=",")[["POLYLINE", "TIMESTAMP"]].values]
train_ = [polyline_to_df(np.array(coord), t, idt) for idt, (boolf, coord, t) in
          enumerate(filter(lambda x: x[0], train_filter))]
train = pd.concat(train_).astype({"occupancy": np.int, "time": np.int, "id_traj": np.int})
nb_traj = len(train.id_traj.unique())
te = time.time()

print("%d Trip extracted in %.1f seconds." %(nb_traj, te-ts))
train.to_pickle(OUTPUT_FILE)

print("File "+ OUTPUT_FILE + " saved.")
