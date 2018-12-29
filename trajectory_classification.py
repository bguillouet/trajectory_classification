import pandas as pd
import pickle
from utils.generate_data import DATA_DIR
from utils.classification import compute_classification_score


print("Caltrain Dataset")

# Download data
data_original = pd.read_pickle(DATA_DIR + "Caltrain.pkl")[["id_traj", "lons", "lats"]].reset_index(drop=True)
nb_traj = len(data_original.id_traj.unique())

#load labels
labels = pickle.load(open(DATA_DIR+"caltrain_traj_labels.pkl", "rb"))


# Load gmm scores
data_score = pd.read_pickle(DATA_DIR+"gmm_scores_Caltrain.pkl")


# Compute trajectory's classification score
pct_good_classification, pct_good_classification_top_3 = compute_classification_score(data_original, data_score, labels, nb_traj)

print("Percentage of well classified trajectory : %.2f" %pct_good_classification )
print("Percentage of of trajectories where good prediction are within best-3 predictions: %.2f \n" %pct_good_classification_top_3 )


## SaoBento
print("SaoBento Dataset")

# Download data
data_original = pd.read_pickle(DATA_DIR + "Sao_bento.pkl")[["id_traj", "lons", "lats"]].reset_index(drop=True)
nb_traj = len(data_original.id_traj.unique())

#load labels
labels = pickle.load(open(DATA_DIR+"sao_bento_traj_labels.pkl", "rb"))


# Load gmm scores
data_score = pd.read_pickle(DATA_DIR+"gmm_scores_saobento.pkl")

# Compute trajectory's classification score
pct_good_classification, pct_good_classification_top_3 = compute_classification_score(data_original, data_score, labels, nb_traj)

print("Percentage of well classified trajectory : %.2f" %pct_good_classification )
print("Percentage of of trajectories where good prediction are within best-3 predictions: %.2f \n" %pct_good_classification_top_3 )

