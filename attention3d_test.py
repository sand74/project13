from transformers import BertModel
from tcv import tcv

model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

fig = tcv.show_3d_output(model, )
fig.show()