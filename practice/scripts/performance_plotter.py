import pandas as pd

from nemd import plotutils

mac = pd.read_csv('~/git/nemd/test/performance/0001/0001_macOS.csv',
                  index_col=0)
ylabel = mac.columns[0]
mac.rename(columns={ylabel: "macOS"}, inplace=True)
ubuntu = pd.read_csv('~/git/nemd/test/performance/0001/0001_ubuntu.csv',
                     index_col=0)
mac.rename(columns={ylabel: "Ubuntu"}, inplace=True)
data = pd.concat((mac, ubuntu), axis=1)
data.sort_index(inplace=True)
with plotutils.pyplot() as plt:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale('log')  # Log scaling on x-axis
    ax.set_yscale('log')  # Log scaling on y-axis
    ax.set_xlabel(data.index.name)
    for column in data.columns:
        ax.plot(data.index,
                data[column],
                marker='o',
                linestyle='--',
                label=column)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')
    # fig.tight_layout()
    fig.savefig(f"performance.png")
