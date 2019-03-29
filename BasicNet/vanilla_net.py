import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        # identity = x
        #print ("identity size", identity.size())
        out = self.conv(x)
        #print ("after conv", out.size())
        out = self.bn(out)
        #print ("after bn", out.size())
        out = self.relu(out)
        #print ("after relu", out.size())
        # residual
        #out += identity
        #out = self.relu(out)

        return out


class BasicNet(nn.Module):

    def __init__(self, zero_init_residual=False):
        super(BasicNet, self).__init__()
        stride = 2
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = BasicBlock(64, 64, stride)
        self.layer2 = BasicBlock(64, 128, stride)
        self.layer3 = BasicBlock(128, 256, stride)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #print ("size of x before going into first conv0", x.size())
        x = self.conv0(x)
        #print ("after conv0", x.size())
        x = self.bn0(x)
        #print ("after bn0", x.size())
        x = self.relu(x)
        #print ("after relu", x.size())
        x = self.maxpool(x)
        #print ("after maxpool", x.size())

        #print ("size of x before going into layer 1 ", x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        fully_connected = x.view(x.size(0), -1)
        #print ("THE FULLY CONNECTED LAYER", x)
        x = self.fc(fully_connected)

        return x, fully_connected


def make_net(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = BasicNet()
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

if __name__ == "__main__":
    net = make_net()
    npd = sorted(dict(net.named_parameters()).items()) 
    param_cnt = 0
    for d in npd:
        print(d[0], d[1].shape, d[1].device) 
        param_cnt += np.prod(d[1].shape)

    print("Model param_cnt: ", param_cnt)
    #print (net)
    imgs = torch.randn(2, 3, 224, 224)
    net(imgs)
    print ("done")



