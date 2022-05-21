# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Global parameters
colors = ['b','r','g','m','y','c']
styles = ['o','s','v','^','D',">"]

def plot_single_perf(bm, df, xaxis, unique_labels):
    fig = fig = plt.figure(1,figsize=(5, 5))
    fig.suptitle(bm)
    
    ax = fig.gca()
    ax.set_xlabel(xaxis)
    ax.set_ylabel('GPU Time (sec)')
    
    ax.set_xscale('log')
    ax.set_xticks(list(df[xaxis]))
    ax.xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    marker_handles = []
        
    num_style = len(df["Distribution"].unique())

    # Iterate over labels and label indices
    for lindex, lbl in enumerate(unique_labels):
        tmpdf = df.loc[df['Label'] == lbl]

        x = tmpdf[xaxis]
        perf = tmpdf["GPU Time (sec)"]

        # Get style & type index
        sid = lindex % num_style
        tid = int(lindex / num_style)

        if not tid:
            ax.plot(x, perf, color=colors[sid])
            ax.scatter(x, perf, color=colors[sid], marker=styles[sid])

            # Add legend
            marker_handles.append(ax.plot([], [], c=colors[sid], marker=styles[sid], \
                                          label=lbl)[0])
        else:
            ax.plot(x, perf, color=colors[sid], linestyle="--")
            ax.scatter(x, perf, color=colors[sid], marker=styles[sid], facecolors='none')

            # Add legend
            marker_handles.append(ax.plot([], [], c=colors[sid], marker=styles[sid], \
                                          mfc='none', linestyle="--", label=lbl)[0])

    leg = plt.legend(handles = marker_handles, loc="upper left", ncol=2, frameon=False)
    plt.savefig(bm + '.eps')

def plot_dual_perf(bm, df, xaxis, unique_labels):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(bm)

    marker_handles = []

    lax = [ax1, ax2, ax3]

    for item in lax:
        item.set_xlabel(xaxis)
        item.set_ylabel("GPU Time (sec)")

    num_style = len(df["Distribution"].unique())

    # Iterate over labels and label indices
    for lindex, lbl in enumerate(unique_labels):
        tmpdf = df.loc[df['Label'] == lbl]

        x = tmpdf[xaxis]
        perf = tmpdf["GPU Time (sec)"]

        # Get style & type index
        sid = lindex % num_style
        tid = int(lindex / num_style)

        # INT32
        if not tid:
            lax[sid].plot(x, perf, color=colors[sid])
            lax[sid].scatter(x, perf, color=colors[sid], marker=styles[sid])

            # Add legend
            marker_handles.append(lax[sid].plot([], [], c=colors[sid], marker=styles[sid], \
                                          label=lbl)[0])
        # INT64
        else:

            lax[sid].plot(x, perf, color=colors[sid], linestyle="--")
            lax[sid].scatter(x, perf, color=colors[sid], marker=styles[sid], facecolors='none')

            # Add legend
            marker_handles.append(lax[sid].plot([], [], c=colors[sid], marker=styles[sid], \
                                          mfc='none', linestyle="--", label=lbl)[0])
    
    leg = plt.legend(handles = marker_handles, loc="upper left", ncol=2, frameon=False)
    plt.savefig(bm + '.eps')