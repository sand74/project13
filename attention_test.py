from transformers import BertModel
import matplotlib.pyplot as plt
from tcv import tcv

model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

fig = tcv.show_output(model, )
fig.show()
