import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_NN(nn.Module):
    def __init__(self, input_dim=26, hidden_dims=[128, 128, 64], output_dim=4, dropout=0.2, activation='relu'):
        """Initializing a linear feed-forward neural network. 
        The linear layers are stored in a ModuleList for dynamical callability. (Number of hidden 
        layers can be set on demand)
        Activation_function = ReLU
        dropout with probability specified in parameter dropout.
        """
        super().__init__()

        # creating a ModuleList for the linear layers:
        self.hidden_layers = nn.ModuleList()

        # hidden layers:
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.hidden_layers.append(nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True))
            else:
                self.hidden_layers.append(nn.Linear(in_features=hidden_dims[i-1], out_features=hidden_dim, bias=True))

        # adding the output layer:
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=output_dim, bias=True)

        # nonlinearity:
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU()

        # adding dropout:
        self.dropout = nn.Dropout(p=dropout)

        # initializing layer weights and biases:
        for i, layer in enumerate(self.hidden_layers):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


        # initializing the output layer:
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        

    def forward(self, x):
        """forward method of the neural network."""

        # hidden layers:
        for layer in self.hidden_layers:
            x = layer(self.dropout(x))
            x = self.act(x)
        
        # output layer:
        x = self.output_layer(self.dropout(x))
        out = F.softmax(x, dim=1)

        return out
