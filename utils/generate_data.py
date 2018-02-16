import pandas as pd
import numpy as np
from traj_dist.cydist.basic_geographical import c_great_circle_distance
import collections

DATA_DIR = "data/"

# Read Poi's Limit
POI_DIR = DATA_DIR + "/POI.csv"
POI_DF = pd.read_csv(POI_DIR, index_col=0)
LON_SB, LAT_SB, TOL_LONS_SB, TOL_LATS_SB = POI_DF.loc["SaoBento"].values
LON_C, LAT_C, TOL_LONS_C, TOL_LATS_C = POI_DF.loc["Caltrain"].values

# Read Area's Limit
SQUARE_DIR = DATA_DIR + "/square.csv"
SQUARE_DF = pd.read_csv(SQUARE_DIR, index_col=0)
LON_INF_P, LON_SUP_P, LAT_INF_P, LAT_SUP_P = SQUARE_DF.loc["porto"].values
LON_INF_SF, LON_SUP_SF, LAT_INF_SF, LAT_SUP_SF = SQUARE_DF.loc["san_francisco"].values


COLUMNS = ["lats", "lons", "occupancy", "time", "id_traj"]

#########
# Porto #
#########

def speed_under_25(l_final):
    n = len(l_final)
    i = 0
    under25 = True
    lons_i1 = l_final[0][0]
    lats_i1 = l_final[0][1]
    while (under25 and i < n):
        lons_i = l_final[i][0]
        lats_i = l_final[i][1]
        d = c_great_circle_distance(lons_i, lats_i, lons_i1, lats_i1)
        lons_i1 = lons_i
        lats_i1 = lats_i
        under25 = d / 15 <= 25
        i = i + 1

    return under25

def filter_data(pl, t):
    pl_split = pl.split("],[")
    boolf = False
    l_final = []
    if 2 < len(pl_split) <= 60:
        loc_start = map(float, pl_split[0][2:].split(","))
        if equal_tol(loc_start[0], LON_SB, TOL_LONS_SB) and equal_tol(loc_start[1], LAT_SB, TOL_LATS_SB):
            loc_end = map(float, pl_split[-1][:-2].split(","))
            if LON_INF_P < loc_end[0] < LON_SUP_P and LAT_INF_P < loc_end[1] < LAT_SUP_P:
                l_final = [loc_start] + [map(float, x.split(",")) for x in pl_split[1:-1]] + [loc_end]
                if speed_under_25(l_final):
                    boolf = True
    return boolf, l_final, t


def polyline_to_df(coord, t, idt, columns = COLUMNS):
    nrow = len(coord)
    lons = coord[:, 0]
    lats = coord[:, 1]
    occupancy = np.hstack((np.repeat(1, nrow - 1), 0))
    time = np.arange(t, t + nrow * 15, 15)
    id_traj = np.repeat(idt, nrow)
    new_traj = pd.DataFrame(np.array([lats, lons, occupancy, time, id_traj]).T, columns=columns)
    return new_traj


#################
# San Francisco #
#################

def as_frame(name):
    """USAGE
    Take the name of a file and transform it to a pandas data frame

    INPUT
    name: name of the file

    OUTPUT
    Data frame corresponding to the file where
    lats= latitudes
    lons= longitudes
    occupancy= (1=occupied, 0=free)
    time= timestamp

    """
    frame = pd.read_csv(name, delimiter=' ', names=['lats', 'lons', 'occupancy', 'time'])

    # Sort data frame according to temporal order order
    frame = frame.sort_values('time')
    # Re-index DataFrame
    i = pd.Index(np.arange(frame.shape[0]))
    frame.index = i

    # Consider unexpired trip as "no-trip"
    occ = frame.occupancy
    if occ.values[-1] == 1:
        frame.loc[frame.index[occ[occ == 0].index[-1] + 1:], "occupancy"] = 0

    return frame


def trajet(data_test, nb_traj=0):
    """USAGE
    return columns pickup,dropoff,trajet

    INPUT
    frame

    OUTPUT
    pickup : true if row i is a pickup
    dropoff: true if row i is a dropoff
    trajet : trajet number if the taxi is full, 0 otherwise

    """
    pick_and_drop = data_test.occupancy.values - np.hstack((0, data_test.occupancy[:-1]))
    pickup = (pick_and_drop == 1)
    dropoff = (pick_and_drop == -1)
    a = np.cumsum(np.array(pick_and_drop == 1, dtype=np.int64)) + nb_traj
    mask_trajet = np.ma.masked_where(np.logical_and(data_test.occupancy == 0, np.logical_not(dropoff)), a)
    traj = np.ma.MaskedArray.filled(mask_trajet, fill_value=0)
    return pickup, dropoff, traj


def deltat_forward_column(frame):
    """USAGE
    return the speed columns corresponding to speed estimation of the vehicle

    INPUT
    frame

    OUTPUT
    speed : the speed column
    """
    deltat = np.hstack((0, frame.time.values[1:] - frame.time.values[0:-1]))
    deltad = np.hstack((0, map(c_great_circle_distance, frame.lons.values[1:], frame.lats.values[1:],
                               frame.lons.values[0:-1], frame.lats.values[0:-1])))
    speed = deltad / deltat
    return speed


def equal_tol(a, b, tol=0.):
    """

    :param a: int or float
    :param b: int or float
    :param tol: int or float
    :return: boolean indicating if abs(a-b) < tol
    """
    rep = abs(a - b) < tol
    return rep

def get_good_trajet(df_traj):

    # Trajectory with more than two locations
    correct_id_traj = set([k for k, v in collections.Counter(df_traj.id_traj).items() if v > 2])

    # Trajectory with Pickup train in Caltrain Area
    df_pickup = df_traj[df_traj["pickup"]][["lons", "lats", "id_traj"]].values
    id_traj_in_caltrain = [int(idt) for lo, la, idt in df_pickup if equal_tol(lo, LON_C, TOL_LONS_C)
                                                                and equal_tol(la, LAT_C, TOL_LATS_C)]
    correct_id_traj = correct_id_traj.intersection(set(id_traj_in_caltrain))

    # Trajectory with speed higher than 25
    incorrect_speed = set(df_traj[df_traj["speed"] > 25].id_traj)

    # Trajectory outside of the box
    end_area = np.logical_and(np.logical_and(df_traj.lons > LON_INF_SF, df_traj.lons < LON_SUP_SF),
                              np.logical_and(df_traj.lats > LAT_INF_SF, df_traj.lats < LAT_SUP_SF))
    incorect_location = set(df_traj.id_traj[np.logical_not(end_area)])

    correct_id_traj = correct_id_traj.difference(incorect_location.union(incorrect_speed))
    return correct_id_traj
