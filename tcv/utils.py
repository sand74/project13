import torch.nn as nn
from dataclasses import dataclass
from collections import namedtuple
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import plotly.graph_objects as go

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))
SAVED_PREFIX = "_saved_"


@dataclass
class Node:
    """
    Dataclass for representing a node structure in a neural network.

    Attributes:
        name (str): The name of the node.
        size (str): The size of the node, which may represent its dimensionality or other characteristics.
        obj (nn.Module): The underlying neural network module associated with this node.
    """
    name: str
    size: str
    obj: nn.Module

    def __str__(self):
        """Returns a string representation of the node, combining its name and size."""
        return self.name + self.size if self.size is not None else f"fn={self.name}"

def get_weight(model: nn, path: str):
    """
    Extracts the weight matrix from a specified layer of the neural network model.

    Args:
        model (nn.Module): The neural network model from which to extract weights.
        path (str): The path to the layer in the format 'layer1.layer2...layerN'.

    Returns:
        torch.Tensor: The weight matrix of the specified layer.
    """
    layer = get_layer(model, path)
    return layer.state_dict()['weight']

def get_bias(model: nn, path: str):
    """
    Extracts the bias matrix from a specified layer of the neural network model.

    Args:
        model (nn.Module): The neural network model from which to extract biases.
        path (str): The path to the layer in the format 'layer1.layer2...layerN'.

    Returns:
        torch.Tensor: The bias matrix of the specified layer.
    """
    layer = get_layer(model, path)
    return layer.state_dict()['bias']

def get_layer(model: nn, path: str):
    """
    Retrieves a specific layer from the neural network model based on the provided path.

    Args:
        model (nn.Module): The neural network model from which to retrieve the layer.
        path (str): The path to the layer in the format 'layer1.layer2...layerN'.

    Returns:
        nn.Module: The specified layer of the model.

    Raises:
        AssertionError: If the path is invalid or if the specified index is out of range.
    """
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
    """
    Displays a histogram of the distribution of values in the given matrix.

    Args:
        matrix (torch.Tensor): The matrix for which to represent the distribution.
        bins (int, optional): The number of bins for the histogram. Defaults to 100.

    Returns:
        go.Figure: A Plotly figure object representing the distribution of matrix values.
    """
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
    """
    Automatically clusters the input data using KMeans clustering.

    Args:
        x (np.ndarray): The input data to be clustered.

    Returns:
        np.ndarray: An array of cluster labels assigned to each data point.
    """
    num_clusters = max(2, int(np.sqrt(len(x))))
    kmeans = KMeans(n_clusters=num_clusters)
    return kmeans.fit_predict(x)

def get_pca(x):
    """
    Performs PCA on the input data to reduce its dimensionality to 2 components.

    Args:
        x (np.ndarray): The input data to be transformed.

    Returns:
        np.ndarray: The transformed data in 2D space.
    """
    pca = PCA(n_components=2)
    return pca.fit_transform(x)