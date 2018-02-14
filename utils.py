import pandas as pd
import numpy as np
from traj_dist.cydist.basic_geographical import c_great_circle_distance
import collections


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


CALTRAIN_LATS = 37.7767
CALTRAIN_LONS = -122.395
TOL_LATS = 0.0005
TOL_LONS = 0.001

LAT_INF = 37.74752
LAT_SUP = 37.8163
LON_INF = -122.45752
LON_SUP = -122.37101000000001


def get_good_trajet(df_traj):

    # Trajectory with more than two locations
    correct_id_traj = set([k for k, v in collections.Counter(df_traj.id_traj).items() if v > 2])

    # Trajectory with Pickup train in Caltrain Area
    df_pickup = df_traj[df_traj["pickup"]][["lons", "lats", "id_traj"]].values
    id_traj_in_caltrain = [int(idt) for lo, la, idt in df_pickup if equal_tol(lo, CALTRAIN_LONS, TOL_LONS)
                                                                and equal_tol(la, CALTRAIN_LATS, TOL_LATS)]
    correct_id_traj = correct_id_traj.intersection(set(id_traj_in_caltrain))

    # Trajectory with speed higher than 25
    incorrect_speed = set(df_traj[df_traj["speed"] > 25].id_traj)

    # Trajectory outside of the box
    end_area = np.logical_and(np.logical_and(df_traj.lons > LON_INF, df_traj.lons < LON_SUP),
                              np.logical_and(df_traj.lats > LAT_INF, df_traj.lats < LAT_SUP))
    incorect_location = set(df_traj.id_traj[np.logical_not(end_area)])

    correct_id_traj = correct_id_traj.difference(incorect_location.union(incorrect_speed))
    return correct_id_traj
