from transformers import BertModel
from tcv import tcv

model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

fig = tcv.attention_3d_matrix(model)
fig.show()