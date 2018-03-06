from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt


def plot_starting_from_trajectories(ax, data, origin, origin_lons, origin_lats, scale_step=0.004):
    maplon = [data.lons.min(), data.lons.max()]
    maplat = [data.lats.min(), data.lats.max()]
    m = Basemap(maplon[0], maplat[0], maplon[1], maplat[1], ax=ax)
    for k, traj in data.groupby("id_traj"):
        x, y = m(traj.lons.values, traj.lats.values)
        m.plot(x, y, marker="None", linestyle="-", color="lightslategrey", alpha=0.3, linewidth=0.35)
        m.plot(x[-1], y[-1], marker=".", color="darkred", alpha=1, markersize=2, linestyle="None")
    m.plot(x[-1], y[-1], marker=".", color="darkred", alpha=1, markersize=4, label="end points", linestyle="None")
    x, y = m(origin_lons, origin_lats)
    m.plot(x, y, marker="o", color="black", markersize=9)
    plt.text(x, y - 0.001, origin, color="black", horizontalalignment="center", verticalalignment="top", fontsize=25)

    m.drawmeridians(np.arange(maplon[0], maplon[1] + scale_step * 2, scale_step * 2), fontsize=16, labelstyle="+/-",
                    labels=[0, 0, 0, 1], linewidth=0, rotation=45)
    m.drawparallels(np.arange(maplat[0], maplat[1] + scale_step * 2, scale_step * 2), labelstyle="+/-", fontsize=16,
                    labels=[1, 0, 0, 0],
                    fmt=lambda x: "%.3f" % x + u'\N{DEGREE SIGN}', linewidth=0)
    plt.legend(fontsize=25, numpoints=1)


def plot_starting_from_trajectories_clusters(ax, data, origin_lons, origin_lats, labels, color_dic, scale_step=0.004):
    maplon = [data.lons.min(), data.lons.max()]
    maplat = [data.lats.min(), data.lats.max()]
    m = Basemap(maplon[0], maplat[0], maplon[1], maplat[1], ax=ax)
    for k, traj in data.groupby("id_traj"):
        color = color_dic[labels[k]]
        x, y = m(traj.lons.values, traj.lats.values)
        m.plot(x, y, marker="None", linestyle="-", color=color, alpha=0.5, linewidth=0.35)
    x, y = m(origin_lons, origin_lats)
    m.plot(x, y, marker="o", color="black", markersize=9)
    m.drawparallels(np.arange(maplat[0], maplat[1] + scale_step * 2, scale_step * 2), labelstyle="+/-", fontsize=16,
                    labels=[1, 0, 0, 0],
                    fmt=lambda x: "%.3f" % x + u'\N{DEGREE SIGN}', linewidth=0)
    m.drawmeridians(np.arange(maplon[0], maplon[1] + scale_step * 2, scale_step * 2), fontsize=16, labelstyle="+/-",
                    labels=[0, 0, 0, 1], linewidth=0, rotation=45)
    ax.set_title("b - Result of Clustering", fontsize=44)


def plot_roc(ax, roc_dic, color_dict, title, nb_tc):
    for tc in range(nb_tc):
        fpr, tpr ,auc_v = roc_dic[tc]
        plt.plot(fpr, tpr, label="%.2f" % auc_v, color=color_dict[tc], linewidth=2, linestyle="-")

    plt.plot([0, 1], [0, 1], linestyle="-.", color="black")
    ax.set_xlabel("False positive rate", fontsize=35)
    ax.set_ylabel("True positive rate", fontsize=35)
    ax.set_xticks(np.arange(0, 1.05, 0.05))
    ax.set_yticks(np.arange(0, 1.05, 0.05))
    xtl = [str(x) if i % 2 == 0 else "" for i, x in enumerate(np.arange(0, 1.05, 0.05))]
    ax.set_xticklabels(xtl)
    ax.set_yticklabels(xtl)
    ax.plot([0, 0], [1, 1], color="k", linestyle="dashed")
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.set_title(title, fontsize=35)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
        tick.label.set_rotation(45)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    if nb_tc == 25:
        plt.legend(fontsize=16, ncol=2, loc=4, bbox_to_anchor=(1.38, 0.2), frameon=False, title="AUC per cluster")
    elif nb_tc == 45:
        plt.legend(fontsize=16, ncol=2, loc=4, bbox_to_anchor=(1.38, -0.01), frameon=False, title="AUC per cluster")
    ax.get_legend().get_title().set_fontsize(22)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
