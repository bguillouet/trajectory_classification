import pandas as pd
from utils.generate_data import DATA_DIR, LON_C, LAT_C, LON_SB, LAT_SB
from utils.plot import plot_starting_from_trajectories, plot_starting_from_trajectories_clusters, plt
import pickle

PLOT_DIR = "/Users/bguillouet/These/trajectory-classification/plot/"
caltrain = pd.read_pickle(DATA_DIR+"caltrain.pkl")
labels_caltrain = pickle.load(open(DATA_DIR+"caltrain_traj_labels.pkl", "rb"))
color_dict_25 = pickle.load(open(DATA_DIR+"Paired_25.pkl", "rb"))

saobento = pd.read_pickle(DATA_DIR+"Sao_bento.pkl")
labels_sao_bento = pickle.load(open(DATA_DIR+"sao_bento_traj_labels.pkl", "rb"))
color_dict_45 = pickle.load(open(DATA_DIR+"Paired_45.pkl", "rb"))

# Print Dataset
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

#Print trajectory cluster
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1)
plot_starting_from_trajectories_clusters(ax, caltrain, LON_C, LAT_C, labels_caltrain, color_dict_25)
plt.savefig(PLOT_DIR + "caltrain_trajectory_clustering.png", dpi=200, bbox_inches="tight")
plt.close()


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1)
plot_starting_from_trajectories_clusters(ax, saobento, LON_SB, LAT_SB, labels_sao_bento, color_dict_45)
plt.savefig(PLOT_DIR + "sao_bento_trajectory_clustering.png", dpi=200, bbox_inches="tight")
plt.close()