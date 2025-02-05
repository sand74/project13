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