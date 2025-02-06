import torch
import torch.nn as nn
from tcv import tcv

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output, last_hidden

model = LSTMModel(input_dim=10, hidden_dim=64, output_dim=2)
input_len = 32
x = torch.rand(input_len, 20, 10)

#tcv.get_embeddings(model, 'fc', x, ['1','2','3','4']*8).show()
tcv.get_embeddings(model, 'fc', x).show()