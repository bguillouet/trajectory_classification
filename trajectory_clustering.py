import fastcluster as fc
import scipy.cluster.hierarchy as sch
from utils.generate_data import DATA_DIR
import numpy as np
import pickle

p_dist = np.load(DATA_DIR + "caltrain_sspd_matrix.npy")
Z = fc.linkage(p_dist, method="ward")
labels =sch.fcluster(Z,25,criterion="maxclust")-1
pickle.dump(labels, open(DATA_DIR+"caltrain_traj_labels.pkl", "wb"))


p_dist = np.load(DATA_DIR + "sao_bento_sspd_matrix.npy")
Z = fc.linkage(p_dist, method="ward")
labels =sch.fcluster(Z,45,criterion="maxclust")-1
pickle.dump(labels, open(DATA_DIR+"sao_bento_traj_labels.pkl", "wb"))