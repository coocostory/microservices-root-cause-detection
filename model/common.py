import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # 创建隐藏层
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, size))
            self.hidden_layers.append(nn.ReLU())
            prev_size = size

        # 创建输出层
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        # 前向传播
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output
