from transformers import BertModel
import matplotlib.pyplot as plt
import tcv

model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

g = tcv.attention_matrix(model)
plt.savefig("a_example.png")
