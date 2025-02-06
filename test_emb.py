from transformers import BertModel
import matplotlib.pyplot as plt
from tcv import tcv

model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
print(tcv.get_weight(model, 'encoder.layer.0-11.attention.self.query'))