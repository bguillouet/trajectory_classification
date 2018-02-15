from utils import as_frame, trajet, deltat_forward_column, get_good_trajet, DATA_DIR
import os
import pandas as pd
import time


INPUT_FILES = map(lambda x: DATA_DIR + '/cabspotting/' + x, os.listdir(DATA_DIR + '/cabspotting/'))
OUTPUT_FILE = DATA_DIR+"Caltrain.pkl"


print("Generate Caltrain dataset from files in " + DATA_DIR + '/cabspotting/ folder.')

ts = time.time()
data_to_merge = []
nb_traj = 0
for i,f_name in enumerate(INPUT_FILES):
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
nb_traj = len(data_caltrain.id_traj.unique())
te = time.time()


print("%d Trip extracted in %.1f seconds." %(nb_traj, te-ts))
data_caltrain.to_pickle(OUTPUT_FILE)
print("File "+ OUTPUT_FILE + " saved.")








