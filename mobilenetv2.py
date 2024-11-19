import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class InvertedResidual(nn.Module):  
    def __init__(self, in_channels, out_channels, stride, expand_ratio):  
        super(InvertedResidual, self).__init__()  
        self.stride = stride  
        self.use_res_connect = (stride == 1 and in_channels == out_channels)  

        # 1x1 Convolution  
        self.conv1 = nn.Conv2d(in_channels, in_channels * expand_ratio, kernel_size=1, stride=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(in_channels * expand_ratio)  
        self.relu = nn.ReLU(inplace=True)  
        
        # Depthwise Convolution  
        self.conv2 = nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size=3, stride=stride,  
                               padding=1, groups=in_channels * expand_ratio, bias=False)  
        self.bn2 = nn.BatchNorm2d(in_channels * expand_ratio)  
        
        # 1x1 Convolution  
        self.conv3 = nn.Conv2d(in_channels * expand_ratio, out_channels, kernel_size=1, stride=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(out_channels)  

    def forward(self, x):  
        out = self.conv1(x)  
        out = self.bn1(out)  
        out = self.relu(out)  
        
        out = self.conv2(out)  
        out = self.bn2(out)  
        out = self.relu(out)  
        
        out = self.conv3(out)  
        out = self.bn3(out)  
        
        if self.use_res_connect:  
            return out + x  
        else:  
            return out  

class MobileNetV2(nn.Module):  
    def __init__(self, num_classes=1000):  
        super(MobileNetV2, self).__init__()  
        
        # Initial 3x3 convolution layer  
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(32)  
        self.relu = nn.ReLU(inplace=True)  
        
        # Define the architecture  
        self.features = nn.Sequential(  
            InvertedResidual(32, 16, stride=1, expand_ratio=1),  
            InvertedResidual(16, 24, stride=2, expand_ratio=6),  
            InvertedResidual(24, 24, stride=1, expand_ratio=6),  
            InvertedResidual(24, 32, stride=2, expand_ratio=6),  
            InvertedResidual(32, 32, stride=1, expand_ratio=6),  
            InvertedResidual(32, 64, stride=2, expand_ratio=6),  
            InvertedResidual(64, 64, stride=1, expand_ratio=6),  
            InvertedResidual(64, 64, stride=1, expand_ratio=6),  
            InvertedResidual(64, 64, stride=1, expand_ratio=6),  
            InvertedResidual(64, 96, stride=1, expand_ratio=6),  
            InvertedResidual(96, 96, stride=1, expand_ratio=6),  
            InvertedResidual(96, 96, stride=1, expand_ratio=6),  
            InvertedResidual(96, 160, stride=2, expand_ratio=6),  
            InvertedResidual(160, 160, stride=1, expand_ratio=6),  
            InvertedResidual(160, 160, stride=1, expand_ratio=6),  
        )  

        # Final layers  
        self.conv2 = nn.Conv2d(160, 320, kernel_size=1, stride=1, padding=0, bias=False)  
        self.bn2 = nn.BatchNorm2d(320)  
        
        self.pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Linear(320, num_classes)  

    def forward(self, x):  
        x = self.conv1(x)  
        x = self.bn1(x)  
        x = self.relu(x)  

        x = self.features(x)  

        x = self.conv2(x)  
        x = self.bn2(x)  
        x = self.relu(x)  

        x = self.pool(x)  
        x = torch.flatten(x, 1)  
        x = self.fc(x)  
        
        return x  

# 生成一个模型实例  
model = MobileNetV2(num_classes=1000)  

# 打印模型结构  
print(model)  

# 测试输入  
x = torch.randn(1, 3, 224, 224)  # 输入为一个batch的224x224 RGB图像  
output = model(x)  
print(output.shape)  # 输出的shape应为 [1, num_classes]