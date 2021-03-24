# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Global parameters
colors = ['b','r','g','m','y','c']
styles = ['o','s','v','^','D',">"]

def plot_perf(bm, df, xaxis, unique_labels, is_multivalue = False, is_singletype = False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(bm)
    
    marker_handles = []
    
    # Frist sub-figure shows performance (#ops/s)
    ax1.set_xlabel(xaxis)
    ax1.set_ylabel('Performance (number of operations per second)')

    # Second sub-figure shows bandwidth
    ax2.set_xlabel(xaxis)
    ax2.set_ylabel('Bandwidth (GB/s)')
    
    # Custom x-axis for multi-value test cases:
    # - power-of-two ticks
    # - log-scale
    lax = [ax1, ax2]
    if is_multivalue:
        lnum = list(df[xaxis])
        
        for item in lax:
            item.set_xscale('log')
            item.set_xticks(lnum)
            item.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        
    if not is_singletype:
        num_style = len(df["Distribution"].unique())
    else:
        num_style = len(df["CGSize"].unique())

    # Iterate over labels and label indices
    for lindex, lbl in enumerate(unique_labels):
        tmpdf = df.loc[df['Label'] == lbl]

        x = tmpdf[xaxis]
        perf = tmpdf["Elem/s (elem/sec)"]
        bw   = tmpdf['Bandwidth (GB/s)']

        # Get style & type index
        sid = lindex % num_style
        tid = int(lindex / num_style)

        if (not tid) or is_singletype:
            ax1.plot(x, perf, color=colors[sid])
            ax1.scatter(x, perf, color=colors[sid], marker=styles[sid])

            ax2.plot(x, bw, color=colors[sid])
            ax2.scatter(x, bw, color=colors[sid], marker=styles[sid])

            # Add legend
            marker_handles.append(ax1.plot([], [], c=colors[sid], marker=styles[sid], \
                                          label=lbl)[0])
        else:
            ax1.plot(x, perf, color=colors[sid], linestyle="--")
            ax1.scatter(x, perf, color=colors[sid], marker=styles[sid], facecolors='none')

            ax2.plot(x, bw, color=colors[sid], linestyle="--")
            ax2.scatter(x, bw, color=colors[sid], marker=styles[sid], facecolors='none')

            # Add legend
            marker_handles.append(ax1.plot([], [], c=colors[sid], marker=styles[sid], \
                                          mfc='none', linestyle="--", label=lbl)[0])
    
    leg = plt.legend(handles = marker_handles, loc="lower left", ncol=2, frameon=False)
    plt.savefig(bm+'.eps')