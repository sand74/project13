import torch.nn as nn

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