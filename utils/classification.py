from .gmm_cv import create_cumsum_data


def compute_classification_score(data_original, data_score, labels, nb_traj):
    """
    USAGE
    Compute classification score of trajectory classification's method

    INPUT
    data_original : pandas DataFrame. Original dataframe with id_traj.
    data_score : pandas DataFrame. Probability to belong to each Gaussian Mixture model for all points.
    labels : list. label of trajector clustering
    nb_traj : int. total number of trajectories

    OUTPUT
    pct_good_classification : float. Percentage of well classified trajectories.
    pct_good_classification_top_3 : float. Percentage of of trajectories where good prediction are within best-3 predictions
    """

    data_cumsum = create_cumsum_data(data_score)

    data_rank = data_cumsum.rank(1, ascending=False)
    data_rank["traj_clust"] = [labels[idt] for idt in data_original.id_traj]
    data_rank["position"] = data_rank.apply(lambda r: r["score_" + str(int(r.traj_clust))], 1)
    data_rank["id_traj"] = data_original.id_traj

    pct_good_classification = float((data_rank.groupby("id_traj").last()["position"] == 1).sum()) / nb_traj * 100
    pct_good_classification_top_3 = float((data_rank.groupby("id_traj").last()["position"] <= 3).sum()) / nb_traj * 100
    return pct_good_classification, pct_good_classification_top_3