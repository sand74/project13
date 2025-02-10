import torch.nn as nn
from dataclasses import dataclass
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import plotly.graph_objects as go


def get_weight(model: nn, path: str):
    """Args:
         model: neural network model.
         path: path for extracting weights in format layer1.layer2. ... .layerN.
       Returns:
         Return weight matrix."""
    layer = get_layer(model, path)
    return layer.state_dict()['weight']

def get_bias(model: nn, path: str):
    """Args:
         model: neural network model.
         path: path for extracting bias in format layer1.layer2. ... .layerN.
       Returns:
         Return bias matrix"""
    layer = get_layer(model, path)
    return layer.state_dict()['bias']

def get_layer(model: nn, path: str):
    """Args:
         model: neural network model.
         path: path in format layer1.layer2. ... .layerN.
       Returns:
         Return current layer as torch.nn."""
    parts = path.split(".")
    layer = model
    for part in parts:
        if part.isdigit():
            part = int(part)
            #assert isinstance(layer, list), f"Error: Layer is not iterable"
            assert 0 <= part < len(layer), f"Error: Index {part} is out of range"
            layer = layer[part]
        else:
            assert hasattr(layer, part), "Error: The wrong path was entered"
            layer = getattr(layer, part)
    return layer

def show_tensor(matrix, bins=100):
    """Args:
         matrix: matrix for representing distribution.
         bins: the number of marks on the graphic.
       Returns:
         Return distribution graphic"""
    pca = PCA(n_components=1, svd_solver='arpack')
    np_matrix = matrix.detach().numpy()
    one_dim = pca.fit_transform(np_matrix).flatten()

    fig = go.Figure(data=[go.Histogram(x=one_dim, nbinsx=bins)])

    fig.update_layout(
        title="Distribution of Matrix Values",
        xaxis_title="PCA Projection",
        yaxis_title="Frequency",
        template="plotly_white"
    )

    return fig

def auto_cluster(x):
    num_clusters = max(2, int(np.sqrt(len(x))))
    kmeans = KMeans(n_clusters=num_clusters)
    return kmeans.fit_predict(x)

def get_pca(x):
    pca = PCA(n_components=2)
    return pca.fit_transform(x)