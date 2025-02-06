import torch
from torch.autograd import Variable
from collections import namedtuple
from transformers import AutoTokenizer
import plotly.graph_objects as go

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
    assert isinstance(var, torch.Tensor), "Error: The 'var' variable should be torch.Tensor"
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

def weight_matrix(model: nn, path: str):
    """Args:
         model: neural network model.
         path: path for extracting weights in format layer1.layer2. ... .layerN.
       Return:
         Graphic of a weight matrix."""
    weight = get_weight(model, path)
    return get_distribution(weight)

def attention_matrix(model: nn, layer: int=0, input: str="Hello, how are you?", tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")):
    """Args:
         model: neural network model.
         layer: number of attention layer.
         input: input text for testing.
         tokenizer: current model tokenizer.
       Return:
         Graphic of an attention matrix."""
    inputs = tokenizer(input, return_tensors="pt")
    outputs = model(**inputs)

    attentions = torch.stack(outputs.attentions)

    # averaging by heads
    attention = attentions[layer][0].mean(axis=0).detach().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return go.Figure(data=go.Heatmap(z=attention, x=tokens, y=tokens, colorscale='Viridis'))

def attention_3d_matrix(model, layer: int=0, input: str="Hello, how are you?", tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")):
    """Args:
         model: neural network model.
         layer: number of attention layer.
         input: input text for testing.
         tokenizer: current model tokenizer.
       Return:
         3D graphic of an attention matrix."""
    inputs = tokenizer(input, return_tensors="pt")
    outputs = model(**inputs)

    attentions = torch.stack(outputs.attentions)

    # averaging by heads
    attention = attentions[layer][0].mean(axis=0).detach().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return go.Figure(data=[go.Surface(z=attention, x=tokens, y=tokens)])

# def sample_characteristic(model: nn, path: str):
#     """Args:
#          model: neural network model.
#          path: path for extracting characteristic in format layer1.layer2. ... .layerN.
#        Returns:
#          Return ????."""
#     pass

def plot_graph(mapa: dict, edges: list[tuple[str]]):
    """
    graph structure plot
    """
    import networkx as nx
    import plotly.graph_objects as go
    from dash import Dash, dcc, html, Output, Input
    names = {}
    for i in mapa:
        names[i] = mapa[i].name
    app = Dash(__name__)
    G = nx.DiGraph()
    G.add_edges_from(edges)
    queue = []
    for i in G.nodes:
        if not any(G.successors(i)):
            queue.append((i, 0))
    levels = {}
    biases = {}
    seen_nodes = set()
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
                if level+1 in levels:
                    levels[level+1].append(neighbor)
                else:
                    levels[level+1] = [neighbor]
    pos = {}
    for level in levels.keys():
        for (bias, node) in enumerate(levels[level]):
            pos[node] = (0.3 * bias, -5 * level)
    
    # Функция для отрисовки DAG
    def create_dag_figure(selected_node=None):
        fig = go.Figure()
    
        # Добавляем рёбра
        for start, end in G.edges():
            x0, y0 = pos[start]
            x1, y1 = pos[end]
            y1 -= 0.5
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
        shapes = []
        annotations = []
        node_labels = []
        for node, (x, y) in pos.items():
            text_width = len(node) * 0.12
            rect_x0, rect_x1 = x - text_width, x + text_width
            rect_y0, rect_y1 = y - 0.5, y + 0.5
    
            shapes.append(dict(type="rect", x0=rect_x0, x1=rect_x1, y0=rect_y0, y1=rect_y1, line=dict(color="black"), fillcolor="lightblue"))
            annotations.append(dict(x=x, y=y, text=node, showarrow=False, font=dict(size=14), xanchor="center", yanchor="middle"))
            
            node_labels.append(node)
    
        fig.update_layout(title="",
                          xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1, max(levels.keys()) + 1]),
                          yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-len(G.nodes), 1]),
                          margin=dict(l=0, r=0, t=0, b=0),
                          plot_bgcolor='white', clickmode="event")
        fig.add_trace(go.Scatter(
            x=[x for x, y in pos.values()],
            y=[y for x, y in pos.values()],
            mode="markers+text",
            marker=dict(size=40, color="lightblue"),
            text=[names[node].replace("Backward", "") for node in pos.keys()],
            textposition="middle center",
            customdata=node_labels,
            hoverinfo="text",
            marker_symbol="square",
            showlegend=False
        ))
    
        
        return fig
    
    def create_right_figure(selected_node=None):
        fig = go.Figure()
        if selected_node:
            fig.add_trace(go.Scatter(
                x=[0], y=[0], text=[mapa[selected_node]], mode="text",
                textfont=dict(size=15, color="black")
            ))
    
        fig.update_layout(title="Выбранная вершина", xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor='white')
    
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
        if clickData and "customdata" in clickData["points"][0]:
            selected_node = clickData["points"][0]["customdata"]
            return create_right_figure(selected_node)
        return create_right_figure()
    app.run_server(debug=False)
