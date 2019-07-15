__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from sklearn.manifold import TSNE
from torchvision.datasets import MNIST

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio

from utils import img_transform


def tsne_plot(X, y, filename=None, backend='plotly', auto_open=False):
    """

    Args:
        X (np.ndarray): 2 dimensional feature array -> size(batch_size, channel*height*widht)
        y (np.ndarray): 1 dimensional corresponding label array -> size(batch_size)
        backend: type of plotting library. Available options are 'plotly' and 'seaborn'

    """
    model = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200, n_iter=1000)
    reduced_data = model.fit_transform(X)

    reduced_df = pd.DataFrame(data={'X': reduced_data[:, 0],
                                    'Y': reduced_data[:, 1],
                                    'label': y})
    if backend == 'plotly':
        data = [go.Scatter(x=group['X'],
                           y=group['Y'],
                           mode='markers',
                           name=label)
                for label, group in reduced_df.groupby('label')]

        layout = go.Layout(
            xaxis={'title': 'X'},
            yaxis={'title': 'Y'}
        )

        fig = go.Figure(data=data, layout=layout)

        pio.write_html(fig, filename)

        py.plot(fig, auto_open=auto_open)


    if backend == 'seaborn':
        g = sns.FacetGrid(reduced_df, hue='label', height=6).map(plt.scatter, 'X', 'Y').add_legend()
        plt.show()


if __name__ == "__main__":
    SUBSET_SIZE = 1000
    dataset = MNIST('../data', transform=img_transform, download=True)

    subset_X = dataset.data[:SUBSET_SIZE].view(-1, 28 * 28).numpy()
    subset_y = dataset.targets.numpy()[:SUBSET_SIZE].astype(int)

    tsne_plot(subset_X, subset_y)

