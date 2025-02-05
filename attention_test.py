from transformers import BertModel
import matplotlib.pyplot as plt
from tcv import tcv

model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

g = tcv.attention_matrix(model, layer=1)
plt.savefig("a_example.png")
