import pandas as pd
import pickle
from utils.generate_data import DATA_DIR
from utils.gmm_cv import build_cross_clust_mixt_cv, compute_classification_score


## Caltrain
print("Caltrain Dataset")

# Download cv_list (for result reproduction)
output_cv = DATA_DIR + "cv_list_Caltrain.pkl"
cv_list = pickle.load(open(output_cv, "r"))

# Download data
data_original = pd.read_pickle(DATA_DIR + "Caltrain.pkl")[["id_traj", "lons", "lats"]].reset_index(drop=True)
nb_traj = len(data_original.id_traj.unique())


# Download trajectory's cluster labels
labels = pickle.load(open(DATA_DIR+"caltrain_traj_labels.pkl", "rb"))
nb_traj_class = labels.max()+1

# Compute gaussian mixture model for all validation dataset and all cluster of trajectory
print("Build Gaussian Mixture Model for all train dataset under 10 Cross-Validation Procedure.")

gmm_cv, data_score = build_cross_clust_mixt_cv(data_original, cv_list, nb_traj_class, labels)
pickle.dump(gmm_cv, open(DATA_DIR + "gmm_cv_models_Caltrain.pkl","wb"))
data_score.to_pickle(DATA_DIR+"gmm_scores_Caltrain.pkl")

# Compute trajectory's classification score
pct_good_classification, pct_good_classification_top_3 = compute_classification_score(data_original, data_score, labels, nb_traj)

print("Percentage of well classified trajectory : %.2f" %pct_good_classification )
print("Percentage of of trajectories where good prediction are within best-3 predictions: %.2f \n" %pct_good_classification_top_3 )




## SaoBento
print("SaoBento Dataset")

# Download cv_list (for result reproduction)
output_cv = DATA_DIR + "cv_list_SaoBento.pkl"
cv_list = pickle.load(open(output_cv, "r"))

# Download data
data_original = pd.read_pickle(DATA_DIR + "Sao_bento.pkl")[["id_traj", "lons", "lats"]].reset_index(drop=True)
nb_traj = len(data_original.id_traj.unique())


# Download trajectory's cluster labels
labels = pickle.load(open(DATA_DIR+"sao_bento_traj_labels.pkl", "rb"))
nb_traj_class = labels.max()+1

# Compute gaussian mixture model for all validation dataset and all cluster of trajectory
print("Build Gaussian Mixture Model for all train dataset under 10 Cross-Validation Procedure.")

gmm_cv, data_score = build_cross_clust_mixt_cv(data_original, cv_list, nb_traj_class, labels)
pickle.dump(gmm_cv, open(DATA_DIR + "gmm_cv_models_saobento.pkl","wb"))
data_score.to_pickle(DATA_DIR+"gmm_scores_saobento.pkl")


# Compute trajectory's classification score
pct_good_classification, pct_good_classification_top_3 = compute_classification_score(data_original, data_score, labels, nb_traj)

print("Percentage of well classified trajectory : %.2f" %pct_good_classification )
print("Percentage of of trajectories where good prediction are within best-3 predictions: %.2f \n" %pct_good_classification_top_3 )




