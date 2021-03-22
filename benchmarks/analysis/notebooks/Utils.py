# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Global parameters
colors = ['b','r','g','m','y','c']
styles = ['o','s','v','^','D',">"]

def plot_insert(bm, df, xaxis, unique_labels, flag = False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(bm)
    
    marker_handles = []
    
    ax1.set_xlabel(xaxis)
    ax1.set_ylabel('Performance (number of operations per second)')

    ax2.set_xlabel(xaxis)
    ax2.set_ylabel('Bandwidth (GB/s)')
    
    lax = [ax1, ax2]
    if flag:
        lnum = list(df[xaxis])
        
        for item in lax:
            item.set_xscale('log')
            item.set_xticks(lnum)
            item.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        
    num_dist = len(df["Distribution"].unique())

    for lindex, lbl in enumerate(unique_labels):
        tmpdf = df.loc[df['Label'] == lbl]

        x = tmpdf[xaxis]
        perf = tmpdf["Elem/s (elem/sec)"]
        bw   = tmpdf['Bandwidth (GB/s)']

        # Get distribution & type index
        did = lindex % num_dist
        tid = int(lindex / num_dist)

        if not tid:
            ax1.plot(x, perf, color=colors[did])
            ax1.scatter(x, perf, color=colors[did], marker=styles[did])

            ax2.plot(x, bw, color=colors[did])
            ax2.scatter(x, bw, color=colors[did], marker=styles[did])

            # Add legend
            marker_handles.append(ax1.plot([], [], c=colors[did], marker=styles[did], \
                                          label=lbl)[0])
        else:
            ax1.plot(x, perf, color=colors[did], linestyle="--")
            ax1.scatter(x, perf, color=colors[did], marker=styles[did], facecolors='none')

            ax2.plot(x, bw, color=colors[did], linestyle="--")
            ax2.scatter(x, bw, color=colors[did], marker=styles[did], facecolors='none')

            # Add legend
            marker_handles.append(ax1.plot([], [], c=colors[did], marker=styles[did], \
                                          mfc='none', linestyle="--", label=lbl)[0])
    
    leg = plt.legend(handles = marker_handles, loc="lower left", ncol=2, frameon=False)
    plt.savefig(bm+'.eps')
