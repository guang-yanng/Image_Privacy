import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from .resnet_v1 import ResNet, Bottleneck

class full_image_score(nn.Module):
    def __init__(self, num_classes = 2, pretrained=True):
        super(full_image_score, self).__init__()
        self.resnet101 = ResNet(Bottleneck, [3, 4, 23, 3])
        
        if pretrained:
            pretrained_model = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
            del pretrained_model['fc.weight']
            del pretrained_model['fc.bias']            
            self.resnet101.load_state_dict(pretrained_model)
        
        self.fc6 = nn.Linear(2048, 4096)
        self.fc7 = nn.Linear(4096, num_classes)
        self.ReLU = nn.ReLU(False)
        self.Dropout = nn.Dropout()

        self._initialize_weights()

    # x1 = union, x2 = object1, x3 = object2, x4 = bbox geometric info
    def forward(self, input_image): 
        x = self.resnet101(input_image)

        x = self.Dropout(x)
        fc6 = self.fc6(x)
        x = self.ReLU(fc6)
        x = self.Dropout(x)
        x = self.fc7(x)

        return x, fc6

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()                         