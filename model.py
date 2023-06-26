import torch
from torchvision.models import resnet50, ResNet50_Weights, ResNet34_Weights
from icecream import ic

class Linear2D(torch.nn.Module):
    def __init__(self, size_in, size_out) -> None:
        super().__init__()
        self.linearLayer = torch.nn.Linear(size_in, size_out)
    
    def forward(self, x):        
        return torch.stack([torch.stack([self.linearLayer(j) for j in torch.unbind(k, dim=2)], dim=2) for k in torch.unbind(x, dim=2)], dim=2)

class CustomModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        shape = 2048
        if size == "50":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=ResNet50_Weights.DEFAULT)
        else:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights=ResNet34_Weights.DEFAULT)
            shape = 512
        self.encoder = torch.nn.Sequential(*list(model.children())[:-2])
        self.classifier = Linear2D(shape, 6)

    def forward(self, x):
        return self.classifier(self.encoder(x))
    
    def get_features(self, x):
        return self.forward(x)
    
    
