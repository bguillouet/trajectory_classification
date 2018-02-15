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
