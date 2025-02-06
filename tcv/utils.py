import torch.nn as nn
from dataclasses import dataclass
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px


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


def get_distribution(matrix, bins=100):
    """Args:
         matrix: matrix for representing distribution.
         bins: the number of marks on the graphic.
       Returns:
         Return distribution graphic"""
    pca = PCA(n_components=1, svd_solver='arpack')
    np_matrix = matrix.detach().numpy()
    one_dim = pca.fit_transform(np_matrix).flatten()
    return px.histogram(x=one_dim, nbins=bins, title="Distribution of Matrix Values",
                       labels={'x': 'PCA Projection', 'y': 'Frequency'},
                       template="plotly_white")
    # pca = PCA(n_components=1, svd_solver='arpack')
    # np_matrix = matrix.detach().numpy()
    # one_dim = pca.fit_transform(np_matrix)
    # sns.set_theme(style="whitegrid")
    # return sns.histplot(one_dim, bins=bins)
