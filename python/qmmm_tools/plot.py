# plot.py

def plot_distances(
    xdata,
    avgs,
    stds,
    labels,
    xlabel='Distance (Å)',
    ylabel='Distance (Å)',
    fontsize=10,
    outfile=None,
    show=False
):
    import numpy as np
    import matplotlib.pyplot as plt

    max_val = np.max(avgs + stds)
    ymax = int(np.ceil(max_val)) + 1

    fig, ax = plt.subplots(
        1, 1,
        figsize=(4, 3.33),
        dpi=300,
        constrained_layout=True
    )

    for j in range(len(labels)):
        ax.errorbar(
            xdata,
            avgs[:, j],
            yerr=stds[:, j],
            label=labels[j]
        )

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(0, ymax)
    ax.legend(
        fontsize=fontsize-1,
        ncol=max(1, len(labels)//2),
        loc='upper center'
    )
    ax.grid(linestyle='--', alpha=0.4)

    if outfile:
        fig.savefig(outfile)

    if show:
        plt.show()

    return fig, ax

