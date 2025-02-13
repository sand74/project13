## Overview

The `tcv` library is designed for visualizing data, particularly in the context of neural networks and PyTorch tensors. It provides a set of functions to build graphs, visualize layer weights, attention matrices, and embeddings, as well as to display directed acyclic graphs (DAGs) using the Dash framework.

## Modules

The library consists of two main modules:

-  `tcv`: The core module containing the primary visualization functions.

-  `utils`: A helper module with additional utility functions.

## Functions

### 1\. `build_graph`

`def build_graph(var: torch.Tensor, params: dict | None = None, show_saved: bool | None = False) -> tuple[dict, list]:`

Builds a graph representation from a PyTorch tensor.

-  **Parameters**:

   -  `var`: The tensor for building the graph.

   -  `params`: (Optional) A dictionary of initial parameters.

   -  `show_saved`: (Optional) A flag to indicate whether to show saved states.

-  **Returns**: A tuple containing the model structure graph as a dictionary and a list of edges.

---

### 2\. `show_layer`

`def show_layer(model: nn, path: str) -> go.Figure:`

Displays the weight distribution of a specified layer in a neural network.

-  **Parameters**:

   -  `model`: The neural network from which to extract weights.

   -  `path`: The path to the layer in the format 'layer1.layer2...layerN'.

-  **Returns**: A graphical representation of the weight matrix of the specified layer.

---

### 3\. `show_output`

`def show_output(model: nn, layer: str, attention_layer_num: int = 0, input_: str = "", tokenizer=None) -> go.Figure:`

Displays a heatmap of the attention matrix for a specified layer in the model.

-  **Parameters**:

   -  `model`: The neural network model from which to extract attention data.

   -  `layer`: The name of the layer containing the attention mechanism.

   -  `attention_layer_num`: (Optional) The index of the attention layer to visualize. Defaults to 0.

   -  `input_`: (Optional) The input text for testing the model. Defaults to "Hello, how are you?".

   -  `tokenizer`: The tokenizer associated with the model. Defaults to the BERT tokenizer.

-  **Returns**: A graphical representation (heatmap) of the attention matrix.

---

### 4\. `show_3d_output`

`def show_3d_output(model: nn, layer: str, attention_layer_num: int = 0, input_: str = "", tokenizer=None) -> go.Figure:`

Displays a 3D surface plot of the attention matrix for a specified layer in the model.

-  **Parameters**: Same as `show_output`.

-  **Returns**: A 3D graphical representation of the attention matrix.

---

### 5\. `get_layer_output`

`def get_layer_output(model: nn, layer: str, input_: transformers.tokenization_utils_base.BatchEncoding):`

Retrieves the output of a specified layer in the model by registering a forward hook.

-  **Parameters**:

   -  `model`: The neural network model from which to extract layer output.

   -  `layer`: The name of the layer whose output is to be retrieved.

   -  `input_`: The input data for the model.

-  **Returns**: The output activations from the specified layer.

---

### 6\. `get_embeddings`

`def get_embeddings(model, layer, input_: transformers.tokenization_utils_base.BatchEncoding, labels=None):`

Retrieves and visualizes the embeddings from a specified layer of the model using PCA.

-  **Parameters**:

   -  `model`: The neural network model from which to extract embeddings.

   -  `layer`: The name of the layer whose embeddings are to be retrieved.

   -  `input_`: The input data for the model.

   -  `labels`: (Optional) Labels for clustering the embeddings.

-  **Returns**: A scatter plot visualizing the embeddings in PCA space.

---

### 7\. `show_graph`

`def show_graph(mapa: dict, edges: list[tuple[str]], minimize_text: bool = True):`

Visualizes a directed acyclic graph (DAG) using the Dash framework.

-  **Parameters**:

   -  `mapa`: A dictionary mapping node identifiers to their corresponding names.

   -  `edges`: A list of tuples representing directed edges between nodes.

   -  `minimize_text`: (Optional) A flag to minimize text display.

-  **Returns**: None. The function runs a Dash web


