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

def plot_graph(edges: list[tuple[str]], names: dict):
    """
    graph structure plot
    """
    import networkx as nx
    import plotly.graph_objects as go
    from dash import Dash, dcc, html, Output, Input
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
            pos[node] = (2 * bias, -5 * level)
    
    # Функция для отрисовки DAG
    def create_dag_figure(selected_node=None):
        fig = go.Figure()
    
        # Добавляем рёбра
        for start, end in G.edges():
            x0, y0 = pos[start]
            x1, y1 = pos[end]
            # y1 -= 0.5
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines+markers",
                marker=dict(
                    symbol="arrow",
                    size=15,
                    angleref="previous",
                ),
                line=dict(color='black', width=2),
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
            text=[names[node] for node in pos.keys()],
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
                x=[0], y=[0], text=[selected_node], mode="text",
                textfont=dict(size=15, color="black")
            ))
    
        fig.update_layout(title="Выбранная вершина", xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor='white')
    
        return fig
    app.layout = html.Div([
        html.Div([
            dcc.Graph(id="dag-graph", figure=create_dag_figure(), style={"height": "90vh"})
        ], style={"width": "100%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([
            dcc.Graph(id="right-graph", figure=create_right_figure(), style={"height": "90vh"})
        ], style={"width": "0%", "display": "inline-block", "verticalAlign": "top"})
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
