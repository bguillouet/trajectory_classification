import pandas as pd
from utils.generate_data import DATA_DIR, LON_C, LAT_C, LON_SB, LAT_SB
from utils.plot import plot_starting_from_trajectories, plt

PLOT_DIR = "/Users/bguillouet/These/trajectory-classification/plot/"
caltrain = pd.read_pickle(DATA_DIR+"caltrain.pkl")
saobento = pd.read_pickle(DATA_DIR+"Sao_bento.pkl")


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1)
plot_starting_from_trajectories(ax, caltrain, "Caltrain", LON_C, LAT_C)
plt.savefig(PLOT_DIR+"caltrain_dataset.png", dpi=200, bbox_inches="tight")
plt.close()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1)
plot_starting_from_trajectories(ax, saobento, "Sao Bento", LON_SB, LAT_SB)
plt.savefig(PLOT_DIR+"sao_bento_dataset.png", dpi=200, bbox_inches="tight")
plt.close()