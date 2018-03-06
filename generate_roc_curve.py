import pandas as pd
import pickle
from utils.generate_data import DATA_DIR
from utils.gmm_cv import create_roc_dict
import matplotlib.pyplot as plt
from utils.plot import plot_roc


PLOT_DIR = "/Users/bguillouet/These/trajectory-classification/plot/"

# Caltrain

data_original = pd.read_pickle(DATA_DIR + "Caltrain.pkl")[["id_traj", "lons", "lats"]].reset_index(drop=True)
data_score = pd.read_pickle(DATA_DIR+"gmm_scores_Caltrain.pkl")
labels = pickle.load(open(DATA_DIR+"caltrain_traj_labels.pkl", "rb"))
color_dict_25 = pickle.load(open(DATA_DIR+"Paired_25.pkl", "rb"))
nb_tc = len(set(labels))
roc_dic = create_roc_dict(data_original, data_score, labels, nb_tc)
title = "Caltrain station, 25 clusters"
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
plot_roc(ax, roc_dic, color_dict_25, title, nb_tc)
plt.savefig(PLOT_DIR+"Caltrain_roc.png", dpi=200, bbox_inches="tight")
plt.close()


# Sao Bento


data_original = pd.read_pickle(DATA_DIR + "Sao_bento.pkl")[["id_traj", "lons", "lats"]].reset_index(drop=True)
data_score = pd.read_pickle(DATA_DIR+"gmm_scores_saobento.pkl")
labels = pickle.load(open(DATA_DIR+"sao_bento_traj_labels.pkl", "rb"))
color_dict_45 = pickle.load(open(DATA_DIR+"Paired_45.pkl", "rb"))
nb_tc = len(set(labels))
roc_dic = create_roc_dict(data_original, data_score, labels, nb_tc)
title = "Sao Bento station, 45 clusters"
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
plot_roc(ax, roc_dic, color_dict_45, title, nb_tc)
plt.savefig(PLOT_DIR+"Saobento_roc.png", dpi=200, bbox_inches="tight")
plt.close()