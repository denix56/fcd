from torch import nn
from torchvision.models import vgg19, vgg19_bn


class VGG19_flex(nn.Module):
    def __init__(self, num_classes=2, num_channels=3):
            super().__init__()
            self.model = vgg19(num_classes=num_classes)
            if num_channels != 3:
                self.model.features = nn.Sequential(
                    nn.Conv2d(num_channels, self.model.features[0].out_channels, kernel_size=3, padding=1),
                    *self.model.features[1:]
                )
            
    def forward(self, x):
        return self.model(x)
        
        
class VGG19_bn_flex(nn.Module):
    def __init__(self, num_classes=2, num_channels=3):
            super().__init__()
            self.model = vgg19_bn(num_classes=num_classes)
            if num_channels != 3:
                self.model.features = torch.nn.Sequential(
                    torch.nn.Conv2d(num_channels, self.model.features[0].out_channels, kernel_size=3, padding=1),
                    *self.model.features[1:]
                )
            
    def forward(self, x):
        return self.model(x)
