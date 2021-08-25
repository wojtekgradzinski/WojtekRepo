#classes of models

from torch import nn 

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        
        layers = []
        
        for i in range(1, hidden_sizes):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        
        
        
    def forward(self, x):
        
        for i in range(len(self.layers-1)):
            x = self.layers[i](x)
            x = nn.ReLU(x)
        x = self.layers[-1](x)
        return self,fc2(x)       
        
        
        return