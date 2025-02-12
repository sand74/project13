import torch
from torch.autograd import Variable
from collections import namedtuple
from transformers import AutoTokenizer
import transformers

import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import numpy as np

from .utils import *

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))
SAVED_PREFIX = "_saved_"

@dataclass
class Node:
    """Dataclass for representing node structure"""
    name: str
    size: str
    obj: nn.Module

    def __str__(self):
        return self.name + self.size if self.size is not None else f"fn={self.name}"

def build_graph(var: torch.Tensor, params: dict | None=None, show_saved: bool | None=False):
    """Args:
         var: tensor for building graph.
         params: dict of initial parameters.
         show_saved: showing saved flag.
       Returns:
         Return model structure graph."""
    assert var is not None, "Error: The 'var' variable should not be None"
    # assert isinstance(var, torch.Tensor), "Error: The 'var' variable should be torch.Tensor"
    assert isinstance(params, dict | None), "Error: The 'params' variable should be dict"
    assert isinstance(show_saved, bool | None), "Error: The 'show_saved' variable should be bool"

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}
    else:
        param_map = {}

    seen = set()

    obj_by_id = dict()
    edges = []

    def size_to_str(size):
        return '(' + ', '.join(['%d' % v for v in size]) + ')'

    def get_var_name(var, name=None):
        if not name:
            name = param_map[id(var)] if id(var) in param_map else ''
        return name

    def get_fn_name(fn):
        name = str(type(fn).__name__)
        return name

    def add_nodes(fn):
        assert not torch.is_tensor(fn)
        if fn in seen:
            return
        seen.add(fn)

        if show_saved:
            for attr in dir(fn):
                if not attr.startswith(SAVED_PREFIX):
                    continue

                val = getattr(fn, attr)
                seen.add(val)
                attr = attr[len(SAVED_PREFIX):]

                if torch.is_tensor(val):
                    edges.append((str(id(fn)), str(id(val))))
                    obj_by_id[str(id(val))] = Node(get_var_name(val, attr), size_to_str(val.size()), val)

                if isinstance(val, tuple):
                    for i, t in enumerate(val):
                        if torch.is_tensor(t):
                            name = attr + '[%s]' % str(i)
                            edges.append((str(id(fn)), str(id(t))))
                            obj_by_id[str(id(t))] = Node(get_var_name(t, name), size_to_str(t.size()), t)

        if hasattr(fn, 'variable'):
            # if grad_accumulator, add the node for .variable
            var = fn.variable
            seen.add(var)
            edges.append((str(id(var)), str(id(fn))))
            obj_by_id[str(id(var))] = Node(get_var_name(var), size_to_str(var.size()), var)

        # add the node for this grad_fn
        obj_by_id[str(id(fn))] = Node(get_fn_name(fn), None, fn)

        # recurse
        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    edges.append((str(id(u[0])), str(id(fn))))
                    add_nodes(u[0])

        if hasattr(fn, 'saved_tensors'):
            for t in fn.saved_tensors:
                seen.add(t)
                edges.append((str(id(t)), str(id(fn))))
                obj_by_id[str(id(t))] = Node(get_var_name(t), size_to_str(t.size()), t)

    def add_base_tensor(var):
        if var in seen:
            return

        seen.add(var)

        assert hasattr(var, 'size'), "Error: the 'var' variable must have the 'size' attribute"
        obj_by_id[str(id(var))] = Node(get_var_name(var), size_to_str(var.size()), var)

        if (var.grad_fn):
            add_nodes(var.grad_fn)
            edges.append((str(id(var.grad_fn)), str(id(var))))

        if var._is_view():
            add_base_tensor(var._base)
            edges.append((str(id(var._base)), str(id(var))))

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    return obj_by_id, edges

def show_layer(model: nn, path: str):
    """Args:
         model: neural network model.
         path: path for extracting weights in format layer1.layer2. ... .layerN.
       Return:
         Graphic of a weight matrix."""
    weight = get_weight(model, path)
    return show_tensor(weight)

def show_output(model: nn, layer: str, attention_layer_num: int=0, input: str="Hello, how are you?", tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")):
    """Args:
         model: neural network model.
         layer: layer with attention.
         attention_layer_num: number of attention layer.
         input: input text for testing.
         tokenizer: current model tokenizer.
       Return:
         Graphic of an attention matrix."""
    inputs = tokenizer(input, return_tensors="pt")

    attentions = torch.stack([get_layer_output(model, layer, inputs)])

    # averaging by heads
    attention = attentions[attention_layer_num][0].mean(axis=0).detach().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return go.Figure(data=go.Heatmap(z=attention, x=tokens, y=tokens, colorscale='Viridis'))

def show_3d_output(model: nn, layer: str, attention_layer_num: int=0, input: str="Hello, how are you?", tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")):
    """Args:
         model: neural network model.
         attention_layer_num: number of attention layer.
         input: input text for testing.
         tokenizer: current model tokenizer.
       Return:
         3D graphic of an attention matrix."""
    inputs = tokenizer(input, return_tensors="pt")

    attentions = torch.stack([get_layer_output(model, layer, inputs)])

    # averaging by heads
    attention = attentions[attention_layer_num][0].mean(axis=0).detach().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return go.Figure(data=[go.Surface(z=attention, x=tokens, y=tokens)])


def get_layer_output(model, layer, input):
    activations = {}

    def hook_fn(module, input0, output0):
        activations["output"] = output0

    get_layer(model, layer).register_forward_hook(hook_fn)

    if isinstance(input, transformers.tokenization_utils_base.BatchEncoding):
        output = model(**input)
    else:
        output = model(input)
    return activations.get("output", None)

def get_embeddings(model, layer, input, labels=None):
    embeddings = get_layer_output(model, layer, input)
    embeddings = embeddings.detach().numpy()
    embeddings_pca = get_pca(embeddings)

    if labels is None:
        labels = auto_cluster(embeddings_pca)

    fig = go.Figure(data=[go.Scatter(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], mode='markers', marker=dict(color=labels))])
    fig.update_layout(title="Embeddings Visualization", xaxis_title="PCA Component 1", yaxis_title="PCA Component 2")
    return fig

def show_graph(mapa: dict, edges: list[tuple[str]], model):
    """
    graph structure plot
    """
    import networkx as nx
    import plotly.graph_objects as go
    from dash import Dash, dcc, html, Output, Input
    from skimage.measure import block_reduce
    from sklearn.decomposition import PCA
    def delong_text(text):
        return "<br>".join([text[i:i+10] for i in range(0, len(text), 10)])
    names = {}
    for i in mapa:
        names[i] = mapa[i].name
    app = Dash(__name__)
    G = nx.DiGraph()
    G.add_edges_from(edges)
    queue = []
    SIZE = 15
    for i in G.nodes:
        if not any(G.successors(i)):
            queue.append((i, 0))
    levels = {}
    biases = {}
    seen_nodes = set()
    heatmap_size = 0.6
    while queue:
        node, level = queue.pop(0)
        if node not in seen_nodes:
            seen_nodes.add(node)
            if level in levels:
                levels[level].append(node)
            else:
                levels[level] = [node]
        for neighbor in G.predecessors(node):
            if neighbor not in seen_nodes:
                seen_nodes.add(neighbor)
                queue.append((neighbor, level + 1))
                if level + 1 in levels:
                    levels[level + 1].append(neighbor)
                else:
                    levels[level + 1] = [neighbor]
    pos = {}
    for level in levels.keys():
        for (bias, node) in enumerate(levels[level]):
            pos[node] = (1 * bias, -1 * level)
    # Функция для отрисовки Structure
    def create_dag_figure(selected_node=None):
        fig = go.Figure()

        # Добавляем рёбра
        for start, end in G.edges():
            x0, y0 = pos[start]
            x1, y1 = pos[end]
            # y1 -= heatmap_size // 2
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines+markers",
                marker=dict(
                    symbol="arrow",
                    size=15,
                    angleref="previous",
                ),
                line=dict(color='black', width=0.7),
                hoverinfo='none',
                showlegend=False
            ))

        scale = max(levels.keys())
        fig.update_layout(title="Structure",
                          xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1, scale + 1]),
                          yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-scale, 1]),
                          margin=dict(l=0, r=0, t=40, b=0),
                          plot_bgcolor='white', clickmode="event")
        text = [delong_text(names[node]) for node in pos.keys()]

        # Добавляем heatmap'ы
        for i in pos:
            try:
                data = get_weight(model, names[i]).detach().numpy()
                if len(data.shape) == 2:
                    x, y = pos[i]
                    size_x, size_y = data.shape[0], data.shape[1]
                    data = block_reduce(data, (size_x // 10, size_y // 10), func=np.max)
                    size_x, size_y = data.shape[0], data.shape[1]
                    fig.add_trace(go.Heatmap(
                        z=data,
                        x=np.linspace(x - heatmap_size/2, x + heatmap_size/2, size_y),
                        y=np.linspace(y - heatmap_size/2, y + heatmap_size/2, size_x),
                        colorscale="Viridis",
                        name=names[i][-8:],
                        showscale=False,
                        hoverinfo='none'
                    ))
            except Exception as e:
                print(e)

        node_labels = list(pos.keys())

        # Добавляем точки(квадратики)
        fig.add_trace(go.Scatter(
            x=[x for x, y in pos.values()],
            y=[y for x, y in pos.values()],
            mode="markers",
            marker=dict(size=50, color="rgba(0, 0, 0, 0.1)"),
            text=text,
            textposition="middle center",
            customdata=node_labels,
            hoverinfo="text",
            marker_symbol="square",
            showlegend=False
        ))
        return fig

    def create_right_figure(selected_node=None):
        fig = go.Figure()
        fig.update_layout(title="Выбранная вершина", xaxis=dict(visible=False), yaxis=dict(visible=False),
                          plot_bgcolor='white')
        if selected_node:
            fig = show_layer(model, names[selected_node])
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            fig.update_layout(title=names[selected_node])
            return fig
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        return fig

    app.layout = html.Div([
        html.Div([
            dcc.Graph(id="dag-graph", figure=create_dag_figure(), style={"height": "90vh"})
        ], style={"width": "60%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([
            dcc.Graph(id="right-graph", figure=create_right_figure(), style={"height": "90vh"})
        ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top"})
    ])

    @app.callback(
        Output("right-graph", "figure"),
        Input("dag-graph", "clickData")
    )
    def update_right_graph(clickData):
        # print(clickData)
        if clickData and 'customdata' in clickData['points'][0]:
            print(clickData['points'][0]['customdata'])
            return create_right_figure(clickData['points'][0]['customdata'])
        return create_right_figure()
    app.run_server(debug=False)

def distill_graph(dct_, borders, remove_back=True):

    def contains_substring(main_string, substrings):
        for substring in substrings:
            if substring in main_string:
                return True
        return False

    def is_bad(id):
        return contains_substring(dct[id].name.lower(), ['grad', 'tback'])

    if not remove_back:
        def is_bad2(id):
            return contains_substring(dct[id].name, ['AddmmBackward0'])
    else:
        def is_bad2(id):
            return contains_substring(dct[id].name, ['Backward'])
    if not remove_back:
        def is_bad2(id):
            return contains_substring(dct[id].name, ['AddmmBackward0'])
    else:
        def is_bad2(id):
            return contains_substring(dct[id].name, ['Backward'])

    graph = dict()
    inv_graph = dict()
    dct = dct_.copy()

    for key, val in dct.items():
        graph[key] = set()
        inv_graph[key] = set()

    for a, b in borders:
        graph[a].add(b)
        inv_graph[b].add(a)

    def delete_v(vertex):
        #print("del", dct[vertex].name)
        children = graph[vertex]
        parents = inv_graph[vertex]

        for parent in parents:
            graph[parent].update(children)
            graph[parent].remove(vertex)

        for child in children:
            inv_graph[child].update(parents)
            inv_graph[child].remove(vertex)

        del graph[vertex]
        del inv_graph[vertex]
        del dct[vertex]

    for vertex, node in dct_.items():
        if is_bad(vertex):
            delete_v(vertex)

    for vertex, node in dct_.items():
        if (node.name.endswith(".weight")):
            base = node.name[:-7]
            #print(base)
            parents_saved = graph[vertex]
            for parent in parents_saved:
                #print(parent)
                dct[parent].name = base
                childs_saved = inv_graph[parent].copy()
                for child in childs_saved:
                    #print("child", dct[child].name)
                    if base in dct[child].name:
                        delete_v(child)

    keys = list(dct.keys()).copy()
    for vertex in keys:
        if is_bad2(vertex):
            delete_v(vertex)


    new_borders = []
    for vertex, childs in graph.items():
        for child in childs:
            new_borders.append((vertex, child))

    return dct, new_borders

