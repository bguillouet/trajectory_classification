import traj_dist.distance as tdist
import pandas as pd
import numpy as np
from utils.generate_data import DATA_DIR
import time

if False :
    print("Caltrain")
    caltrain = pd.read_pickle(DATA_DIR + "caltrain.pkl")
    output_file = DATA_DIR + "caltrain_sspd_matrix.npy"
    print("Convert Data to list")
    traj_list = [c[1][["lons", "lats"]].values for c in caltrain.groupby("id_traj")]
    ts = time.time()
    p_dist = tdist.pdist(traj_list, metric="sspd")
    nb_dist = len(p_dist)
    te = time.time()
    print("%d Distances computed in %d seconds" % (nb_dist, te-ts))
    np.save(output_file, p_dist)

print("Sao Bento")
sao_bento = pd.read_pickle(DATA_DIR + "sao_bento.pkl")
output_file = DATA_DIR + "sao_bento_sspd_matrix.npy"
print("Convert Data to list")
traj_list = [c[1][["lons", "lats"]].values for c in sao_bento.groupby("id_traj")]
ts = time.time()
p_dist = tdist.pdist(traj_list, metric="sspd")
nb_dist = len(p_dist)
te = time.time()
print("%d Distances computed in %d seconds" % (nb_dist, te-ts))
np.save(output_file, p_dist)
