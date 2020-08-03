from torch import nn
import torch
from torch.functional import F


config = {
    "epoch": 30,
    "batch": 8,
    "conv1_ch": 64,
    "conv2_ch": 64,
    "conv3_ch": 128,
    "fc1_out": 2048,
}

class NeuralNet(nn.Module):
    def __init__(self, input_shape=(1, 10, 40, 30)):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv3d(1, config["conv1_ch"], 3, padding=1)
        self.bn3 = nn.BatchNorm3d(config["conv1_ch"])

        self.conv2 = nn.Conv3d(config["conv1_ch"], config["conv2_ch"], 3, padding=1)
        self.bn4 = nn.BatchNorm3d(config["conv2_ch"])

        self.conv3 = nn.Conv3d(config["conv2_ch"], config["conv3_ch"], 3, padding=1)
        self.bn5 = nn.BatchNorm3d(config["conv3_ch"])

        self.conv4 = nn.Conv3d(config["conv3_ch"], config["conv3_ch"], 3, padding=1)
        self.bn6 = nn.BatchNorm3d(config["conv3_ch"])

        self.pool1 = nn.MaxPool3d(2)

        n_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_size , config["fc1_out"])
        self.fc2 = nn.Linear(config["fc1_out"], config["fc1_out"])
        self.fc3 = nn.Linear(config["fc1_out"], 1)

        self.bn1 = nn.BatchNorm1d(config["fc1_out"])
        self.bn2 = nn.BatchNorm1d(config["fc1_out"])
        self.dropout = nn.Dropout(0.50)

    def _get_conv_output(self, shape):
      batch_size = 1
      input = torch.autograd.Variable(torch.rand(batch_size, *shape))
      output_feat = self._forward_features(input)
      n_size = output_feat.data.view(batch_size, -1).size(1)
      return n_size

    def _forward_features(self, x):
      x = F.relu(self.bn3(self.conv1(x)))
      x = F.relu(self.bn4(self.conv2(x)))
      x = self.pool1(x)
      x = F.relu(self.bn5(self.conv3(x)))
      x = F.relu(self.bn6(self.conv4(x)))
      x = self.pool1(x)
      return x

    def forward(self,x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


def get_model(path, pretrained=True):
    model = NeuralNet()
    if pretrained:
       model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model
