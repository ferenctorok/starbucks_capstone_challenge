import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_NN(nn.Module):
    def __init__(self, input_dim=26, hidden_dims=[128, 128, 64], output_dim=4, dropout=0.2):
        """Initializing a linear feed-forward neural network. 
        The linear layers are stored in a ModuleList for dynamical callability. (Number of hidden 
        layers can be set on demand)
        Activation_function = ReLU
        dropout with probability specified in parameter dropout.
        """
        super().__init__()

        # creating a ModuleList for the linear layers:
        self.layers = nn.ModuleList()

        # hidden layers:
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.layers.append(nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True))
            else:
                self.layers.append(nn.Linear(in_features=hidden_dims[i-1], out_features=hidden_dim, bias=True))

        # adding the output layer:
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=output_dim, bias=True)

        # adding dropout:
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """forward method of the neural network."""

        # hidden layers:
        for layer in self.layers:
            x = layer(self.dropout(x))
            x = F.relu(x)
        
        # output layer:
        x = self.output_layer(self.dropout(x))
        out = F.softmax(x)

        return out
