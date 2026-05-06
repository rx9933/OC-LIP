
import torch
import torch.nn as nn

class GenericDenseSkipLearn0(nn.Module):
    def __init__(self, input_dim=50, hidden_layer_dim=256, output_dim=20, dropout_prob=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Main layers
        self.hidden1 = nn.Linear(input_dim, hidden_layer_dim)
        self.act1 = nn.GELU()
        
        self.hidden2 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.act2 = nn.GELU()
        
        self.hidden3 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.act3 = nn.GELU()
        
        self.hidden4 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.act4 = nn.GELU()
        
        self.output = nn.Linear(hidden_layer_dim, output_dim)
        
        # Skip connection adapter
        self.skip_adapter = nn.Linear(2, output_dim)
        
        # Learnable weights
        self.skip_weight1 = nn.Parameter(torch.tensor(0.5))
        self.skip_weight2 = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x, expected=None):  # Add expected parameter with default None
        # Main path with dropout
        main = self.dropout(self.act1(self.hidden1(x)))
        main = self.dropout(self.act2(self.hidden2(main)))
        main = self.dropout(self.act3(self.hidden3(main)))
        main = self.dropout(self.act4(self.hidden4(main)))
        main = self.output(main)
        
        # Skip connection
        x_first_two = x[:, :2]
        skip = self.skip_adapter(x_first_two)
        
        # Combine
        output = main + self.skip_weight1 * skip + self.skip_weight2 * skip
        
        return output