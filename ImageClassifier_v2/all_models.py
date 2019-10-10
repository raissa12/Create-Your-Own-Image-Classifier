import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F

def AllModel(arch):
    if arch == 'alexnet':
        return models.alexnet(pretrained=True)
    elif arch == 'resnet18':
        return models.resnet18(pretrained=True)
    elif arch == 'resnet34':
        return models.resnet34(pretrained=True)
    elif arch == 'resnet50':
        return models.resnet50(pretrained=True)
    elif arch == 'resnet101':
        return models.resnet101(pretrained=True)
    elif arch == 'resnet152':
        return models.resnet152(pretrained=True)
    elif arch == 'squeezenet1_0':
        return models.squeezenet1_0(pretrained=True)
    elif arch == 'squeezenet1_1':
        return models.squeezenet1_0(pretrained=True)
    elif arch == 'vgg11':
        return models.vgg11(pretrained=True)
    elif arch == 'vgg13':
        return models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        return models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        return models.vgg19(pretrained=True)
    elif arch == 'densenet121':
        return models.densenet121(pretrained=True)
    elif arch == 'densenet161':
        return models.densenet161(pretrained=True)
    elif arch == 'densenet169':
        return models.densenet169(pretrained=True)
    elif arch == 'densenet201':
        return models.densenet201(pretrained=True)
    elif arch == 'inception_v3':
        return models.inception_v3(pretrained=True)
    else:
        print("Error, invalid architecture - ",arch)
        exit()
        return 0
    
# Define custom network architecture
class Classifier(nn.Module):
    def __init__(self,hidden_units):
        super().__init__()
        self.hidden_units = hidden_units
        for idx,layer in enumerate(hidden_units):
            i = str(idx+1)
            if idx == 0:
                setattr(self, 'fc'+i, nn.Linear(25088,layer))
            else:
                setattr(self, 'fc'+str(idx+1), nn.Linear(hidden_units[idx-1],layer))
        setattr(self, 'fc'+str(len(hidden_units)+1), nn.Linear(hidden_units[-1],102))

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        for idx,layer in enumerate(self.hidden_units):
            i = str(idx+1)
            tmp = getattr(self,'fc'+i)
            x = F.relu(tmp(x))
        tmp = getattr(self,'fc'+str(len(self.hidden_units)+1))
        x = F.log_softmax(tmp(x), dim=1)

        return x