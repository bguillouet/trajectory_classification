from utils import as_frame, trajet, deltat_forward_column, get_good_trajet
import os
import pandas as pd



DATA_DIR = "/Users/bguillouet/These/trajectory-classification/data/"
INPUT_FILES = map(lambda x: DATA_DIR + '/cabspotting/' + x, os.listdir(DATA_DIR + '/cabspotting/'))

data_to_merge = []

nb_traj = 0
for i,f_name in enumerate(INPUT_FILES):
    if i%50==0:
         print(i)
    # Load Data
    df = as_frame(f_name)

    # Detect Trip
    pickup, dropoff, traj = trajet(df, nb_traj)
    df["pickup"] = pickup
    df["dropoff"] = dropoff
    df['id_traj'] = traj

    #Compute Speed
    speed = deltat_forward_column(df)
    df["speed"] = speed

    # Remove non-trip data
    df_traj = df[df.id_traj != 0]

    nb_traj+=len(set(traj))-1

    # Remove trip which are not in the desired area
    correct_id_traj = get_good_trajet(df_traj)
    df_good_traj = df_traj[df_traj.id_traj.isin(correct_id_traj)]


    data_to_merge.append(df_good_traj)

data_caltrain = pd.concat(data_to_merge)
new_id_dict = dict([(v,k) for k,v in enumerate(data_caltrain.id_traj.unique())])

data_caltrain["id_traj"] = [new_id_dict[idt] for idt in data_caltrain["id_traj"]]

data_caltrain.to_pickle(DATA_DIR+"Caltrain.pkl")








