import paddle
from paddle import nn
from paddle.nn import functional as F


class DepthWise_Conv(nn.Layer):
    """通道分离卷积-group==in_channels
    """
    def __init__(self,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=0):
        super(DepthWise_Conv, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=in_channels,
                              groups=in_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        
    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Please make sure then depthwise inputs'shape[0]" + \
                "(now:{0}) is equal to the conv in_channels(now:{1}).".format(
                                            inputs.shape[0], self.in_channels)

        x = self.conv(inputs)
        return x

class PointWise_Conv(nn.Layer):
    """逐点卷积-1x1
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(PointWise_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 1
        self.stride = 1
        self.padding = 0

        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0)
    
    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Please make sure then depthwise inputs'shape[0]" + \
                "(now:{0}) is equal to the conv in_channels(now:{1}).".format(
                                            inputs.shape[0], self.in_channels)

        x = self.conv(inputs)
        return x


class Linear_Depth_Separ_Conv(nn.Layer):
    """线性可分离卷积-末尾逐点卷积为线性输出
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.ReLU6):
        super(Linear_Depth_Separ_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.depthwise_conv = DepthWise_Conv(in_channels=in_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding)
        
        self.pointwise_conv = PointWise_Conv(in_channels=in_channels,
                                             out_channels=out_channels)
        
        self.act = act()
        self.depthwise_bn = nn.BatchNorm2D(in_channels)
        self.pointwise_bn = nn.BatchNorm2D(out_channels)
    
    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Please make sure then depthwise inputs'shape[0]" + \
                "(now:{0}) is equal to the conv in_channels(now:{1}).".format(
                                            inputs.shape[0], self.in_channels)

        x = self.depthwise_conv(inputs)
        x = self.act(self.depthwise_bn(x))
        x = self.pointwise_conv(x)
        x = self.act(self.pointwise_bn(x))
        return x

class BottleNeck(nn.Layer):
    """MobileNet倒残差瓶颈块-残差低空间
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 t=1,
                 n=1,
                 kernel_size=3,
                 stride=1,
                 padding=0):
        super(BottleNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t = t
        self.n = n
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.bottle_stem = nn.Sequential(
            PointWise_Conv(in_channels=in_channels,
                           out_channels=t*in_channels),
            nn.BatchNorm2D(t*in_channels),
            nn.ReLU6(),
            Linear_Depth_Separ_Conv(in_channels=t*in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)
        )

        self.bottle_blocks = []
        for i in range(0, n-1):
            self.bottle_blocks.append(
                nn.Sequential(
                    PointWise_Conv(in_channels=out_channels,
                                   out_channels=t*out_channels),
                    nn.BatchNorm2D(t*out_channels),
                    nn.ReLU6(),
                    Linear_Depth_Separ_Conv(in_channels=t*out_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            padding=1) # keep feature map size
                )
            )
        self.bottle_blocks = nn.LayerList(self.bottle_blocks)

    
    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Please make sure then depthwise inputs'shape[0]" + \
                "(now:{0}) is equal to the conv in_channels(now:{1}).".format(
                                            inputs.shape[0], self.in_channels)
                                            
        x = self.bottle_stem(inputs)
        
        for b in self.bottle_blocks:
            x = b(x) + x

        return x


class Stem(nn.Layer):
    """渐入层
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.ReLU6):
        super(Stem, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.act = act()
        self.bn = nn.BatchNorm2D(out_channels)


    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Please make sure then depthwise inputs'shape[0]" + \
                "(now:{0}) is equal to the conv in_channels(now:{1}).".format(
                                            inputs.shape[0], self.in_channels)

        x = self.conv(inputs)
        x = self.act(self.bn(x))
        return x


class Classifier_Head(nn.Layer):
    """分类头
    """
    def __init__(self,
                 in_channels,
                 num_classes):
        super(Classifier_Head, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.conv = PointWise_Conv(in_channels=in_channels, out_channels=num_classes)
        self.act = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Please make sure then depthwise inputs'shape[0]" + \
                "(now:{0}) is equal to the conv in_channels(now:{1}).".format(
                                            inputs.shape[0], self.in_channels)

        x = self.avg_pool(inputs)
        x = self.conv(x)
        x = self.flatten(self.act(x))
        return x
        


class MobileNetV2(nn.Layer):
    """MobileNetV2实现
        Params Info:
            num_classes: 分类数
            in_channels: 输入图像通道数
            alpha: 模型伸缩大小(0.0, 1.0), 建议值:1.0, 0.75, 0.5, 0.25
    """
    def __init__(self,
                 num_classes=1000,
                 in_channels=3,
                 alpha=1.0):
        super(MobileNetV2, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.alpha = alpha

        self.net_channels = [
            # net channels config:
            # in_channels, out_channels,
            # stride, padding,
            # t, n
            [3, int(alpha*32), 2, 1, 0, 1],    # Stem Block --> t = 0, n=1
            [int(alpha*32), int(alpha*16), 1, 1, 1, 1],    # first bottleneck --> t = 1, n=1
            [int(alpha*16), int(alpha*24), 2, 1, 6, 2],    # second bottleneck --> t = 6, n=2
            [int(alpha*24), int(alpha*32), 2, 1, 6, 3],    # first bottleneck --> t = 6, n=3
            [int(alpha*32), int(alpha*64), 2, 1, 6, 4],    # first bottleneck --> t = 6, n=4
            [int(alpha*64), int(alpha*96), 1, 1, 6, 3],    # first bottleneck --> t = 6, n=3
            [int(alpha*96), int(alpha*160), 2, 1, 6, 3],    # first bottleneck --> t = 6, n=3
            [int(alpha*160), int(alpha*320), 1, 1, 6, 1],    # first bottleneck --> t = 6, n=1
            [int(alpha*320), int(alpha*1280), 1, 1, 0, 1],    # end Conv block --> t = 0, n=1
        ]

        self.stem = Stem(in_channels=self.net_channels[0][0],
                         out_channels=self.net_channels[0][1],
                         kernel_size=3,
                         stride=self.net_channels[0][2],
                         padding=self.net_channels[0][3])
        
        self.mb_blocks = []
        for i in range(1, len(self.net_channels)-1):
            self.mb_blocks.append(
                BottleNeck(in_channels=self.net_channels[i][0],
                           out_channels=self.net_channels[i][1],
                           t=self.net_channels[i][4],
                           n=self.net_channels[i][5],
                           kernel_size=3,
                           stride=self.net_channels[i][2],
                           padding=self.net_channels[i][3])
            )
        self.mb_blocks = nn.LayerList(self.mb_blocks)

        self.end_conv = nn.Sequential(
            PointWise_Conv(in_channels=self.net_channels[-1][0],
                           out_channels=self.net_channels[-1][1]),
            nn.BatchNorm2D(self.net_channels[-1][1]),
            nn.ReLU6()
        )

        self.head = Classifier_Head(in_channels=self.net_channels[-1][1],
                                    num_classes=num_classes)

    def forward(self, inputs):
        x = self.stem(inputs)

        for b in self.mb_blocks:
            x = b(x)

        x = self.end_conv(x)
        x = self.head(x)

        return x


def MobileNetV2_for_224(num_classes,
                   in_channels):
    """构建最适合224大小图像的MobileNetV2
    """
    return MobileNetV2(num_classes=num_classes,
                       in_channels=in_channels,
                       alpha=1.4)


def MobileNetV2_Base(num_classes,
                   in_channels):
    """构建基础(大型)MobileNetV2
    """
    return MobileNetV2(num_classes=num_classes,
                       in_channels=in_channels,
                       alpha=1.0)

def MobileNetV2_0_75(num_classes,
                   in_channels):
    """构建较大MobileNetV2
    """
    return MobileNetV2(num_classes=num_classes,
                       in_channels=in_channels,
                       alpha=0.75)

def MobileNetV2_Mid(num_classes,
                   in_channels):
    """构建中等MobileNetV2
    """
    return MobileNetV2(num_classes=num_classes,
                       in_channels=in_channels,
                       alpha=0.5)

def MobileNetV2_Small(num_classes,
                   in_channels):
    """构建最小MobileNetV2
    """
    return MobileNetV2(num_classes=num_classes,
                       in_channels=in_channels,
                       alpha=0.25)
