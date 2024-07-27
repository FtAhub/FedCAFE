import torch
from torch import nn
from torch.nn import functional as F
from models.basic_module import BasicModule

LAYER1_NODE = 8192


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)


class TxtModule(BasicModule):
    def __init__(self, y_dim, bit, num_prototypes):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TxtModule, self).__init__()
        self.module_name = "text_model"

        # full-conv layers
        self.conv1 = nn.Conv2d(1, LAYER1_NODE, kernel_size=(y_dim, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(LAYER1_NODE, bit, kernel_size=1, stride=(1, 1))
        self.apply(weights_init)

        # clissifier
        self.categorizer = nn.Linear(in_features=bit, out_features=num_prototypes)
        self.categorizer.weight.data = torch.randn(num_prototypes, bit) * 0.01
        self.categorizer.bias.data = torch.randn(num_prototypes) * 0.01
        self.assignments = nn.Softmax(dim=1)

    def forward(self, x, prototypes):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x.squeeze()

        # storing raw feature
        raw_feature = x
        predicted_label = self.categorizer(raw_feature)
        predicted_label = self.assignments(predicted_label)

        # computing memory_feature
        memory_feature = torch.matmul(predicted_label, prototypes)

        # computing adaptive selector
        adaptive_selector = nn.Tanh()(raw_feature)

        # enhanced feature
        enhanced_feature = raw_feature + adaptive_selector * memory_feature

        return enhanced_feature, predicted_label, raw_feature

