import torch.nn as nn
import torch.nn.functional as F


#Adjusts channel dimensions; No padding -> spatial dimensions shrink if stride > 1
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


#Used for feature extraction; Padding=1 ensures same spatial size
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride) #3x3 convolution that can downsample if stride>1
        self.conv2 = conv3x3(planes, planes)    #another 3x3 convolution, same numbeer of channels
        self.bn1 = nn.BatchNorm2d(planes)   #batch normalization layers after each conv
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)   #activation function

        #If stride > 1, the input shortcut x needs to be resized (so it matches the shape of y)
        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        #Input x goes through the first conv -> batchnorm -> ReLU
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        #Then through the second conv -> batchnorm; At this stage, y is the "residual mapping"
        y = self.bn2(self.conv2(y))

        #If needed, x is doensampled so its dimensions match y
        if self.downsample is not None:
            x = self.downsample(x)

        #Add input (x) + output (y) -> residual connection (skip connection); Apply final ReLU
        return self.relu(x+y)


#This class is building a hybrid between ResNet and FPN (Feature Pyramid Network)
class ResNetFPN_8_2(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock  #Previously defined
        initial_dim = config['initial_dim'] #Number of channels in the very first layer
        block_dims = config['block_dims']   #List of channel sizes for each ResNet stage

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        #First conv layer: 7x7 kernel, stride = 2 (so resolution -> 1/2); Input has 1 channel
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        #BatchNorm + ReLU standard
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        #Multi-scale feature maps from the backbone
        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # Keeps resolution (1/2)
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # Downsamples (1/4)
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # Downsamples again (1/8)

        # 3. FPN upsample; These are 1x1 convolutions to align channel dimensions before fusion; 
        # Works by taking higher-level (low-res) features and upsampling them, then fusing with lower-level (high-res) features
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        #Upsampled layer3 + processed layer2 -> refined with conv3x3s
        #Output goes back to block_dims[1] channels
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        #Same idea: fuse refined layer2 with layer1, then refine; Output goes back to block_dims[0] channels
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        #Convs: Kaiming initialization (good for ReLU); BatchNorm: set scale=1, bias=0 - starts as identify transform
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    #Creates a stack of two BasicBlocks
    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)  #first block, may downsample (if stride = 2)
        layer2 = block(dim, dim, stride=1)  #second block, keeps resolution fixed
        layers = (layer1, layer2)

        #Update so the next layer knows its input channel count
        self.in_planes = dim
        #Wraps the blocks in an nn.Sequential container
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x))) #after initial 7x7  conv + BN + ReLU, resolution = 1/2
        x1 = self.layer1(x0)  # 1/2, after layer1
        x2 = self.layer2(x1)  # downsampled to 1/4, after layer2
        x3 = self.layer3(x2)  # downsampled to 1/8, after layer3

        # FPN
        #Start from coarsest (1/8); high-level semantics
        x3_out = self.layer3_outconv(x3)    #keep channels consistent

        #Upsample x3_out -> 1/4 resolution
        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        #Add x2 (skip connection from backbone)
        x2_out = self.layer2_outconv(x2)
        #Refine with layer2_outconv2; Result: x2_out, fused 1/4 features
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        #Unsample x2_out -> 1/2
        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        #Add to x1 (skip connection from backbone)
        x1_out = self.layer1_outconv(x1)
        #Refine with layer1_outconv2; Result: x1_out, fused 1/2 features
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        #Returns two scales: x3_out= 1/8 resolution, semantically strong; x1_out= 1/2 resolution, spatially precise
        return [x3_out, x1_out]


class ResNetFPN_16_4(nn.Module):
    """
    ResNet+FPN, output resolution are 1/16 and 1/4.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim'] #Number of channels after the very first convolution
        block_dims = config['block_dims']   #Channels sizes per stage

        # Class Variable; Tracks current channel depth between stages
        self.in_planes = initial_dim

        # Networks
        #7x7 conv (stride = 2) -> reduces resolution to 1/2; Works with one channel (grayscale)
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # keeps resolution (1/2)
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # downsamples to (1/4)
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # downsamples (1/8)
        self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # downsamples to (1/16)

        # 3. FPN upsample
        #Normalizes channels at 1/16 scale
        self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
        #Upsample 1/16 -> fuse with 1/8
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
        #Refine with 3x3 convolutions -> output 1/8 features (block_dims[2] channels)
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[3], block_dims[3]),
            nn.BatchNorm2d(block_dims[3]),
            nn.LeakyReLU(),
            conv3x3(block_dims[3], block_dims[2]),
        )

        #Upsample 1/8 -> fuse with 1/4
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        #Refine with 3x3 convolutions -> output 1/4 features (block_dims[1] channels)
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )

        #Standard Kaiming init for convs, identity init for BN
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        #First block may downsample (if stride > 1)
        layer1 = block(self.in_planes, dim, stride=stride)
        #Second block keeps resolution
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        #Updates self.in_planes for next stage
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16

        # FPN; Start from deepest (1/16)
        x4_out = self.layer4_outconv(x4)

        #Upsample x4 (1/16 -> 1/8)
        x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)
        #Fuse with x3
        x3_out = self.layer3_outconv(x3)
        #Refine with layer3_outconv2; Output: 1/8 features
        x3_out = self.layer3_outconv2(x3_out+x4_out_2x)

        #Upsample x3_out (1/8 -> 1/4)
        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        #Fuse with x2
        x2_out = self.layer2_outconv(x2)
        #Refine with layer2_outconv2; Ouput: 1/4 features
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        #Returns 1/16 (x4_out) and 1/4 (x2_out)
        return [x4_out, x2_out]