import paddle
from paddle import nn
from paddle.nn import functional as F


class DepthWise_Conv(nn.Layer):

    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0):
        super(DepthWise_Conv, self).__init__()
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=in_channels,
                              groups=in_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class PointWise_Conv(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(PointWise_Conv, self).__init__()
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class SEBlock(nn.Layer):

    def __init__(self,
                 in_channles,
                 reduce=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.conv1 = PointWise_Conv(in_channels=in_channles,
                                    out_channels=in_channles // reduce)
        self.conv2 = PointWise_Conv(in_channels=in_channles // reduce,
                                    out_channels=in_channles)
        
        self.act1 = nn.ReLU()
        self.act2 = nn.Hardsigmoid()
    
    def forward(self, inputs):
        x = self.avg_pool(inputs)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = inputs * x # origin input fuses the attention information
        return x


class MLP(nn.Layer):

    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features=None,
                 drop_rate=0.,
                 act=nn.GELU):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = in_features if out_features is None else out_features

        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=self.out_features)

        self.dropout = nn.Dropout(p=drop_rate)
        self.act = nn.GELU()
    
    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DY_ReLU(nn.Layer):
    """动态ReLU-B型:
    """
    def __init__(self,
                 channels,
                 m=None,
                 k=2,
                 use_mlp=False,
                 coefs=[1.0, 0.5],
                 const=[1.0, 0.0],
                 dy_reduce=6):
        super(DY_ReLU, self).__init__()
        self.k = k
        self.m = m
        self.channels = channels
        self.use_mlp = use_mlp
        self.coefs = coefs
        self.const = const

        self.mid_channels = 2*k*channels # 最终映射大小

        # 超参-1: 
        self.lambda_a_coef = coefs[0]
        self.lambda_b_coef = coefs[1]

        # 超参-2: 
        self.lambda_a_const = const[0]
        self.lambda_b_const = const[1]

        # 构建超参
        self.lambda_a_b = paddle.to_tensor([self.lambda_a_coef]*k+\
                                           [self.lambda_b_coef]*k)
        self.init_value_for_a_b = paddle.to_tensor([self.lambda_a_const]+\
                                           [self.lambda_b_const]*(2*k-1))
        
        # 构建动态参数project层
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1) if m is None else nn.AdaptiveAvgPool1D(output_size=1)
        self.project = nn.Sequential(
            MLP(channels if m is None else m, channels//dy_reduce if channels//dy_reduce != 0 else 1) if use_mlp \
                else nn.Linear(channels if m is None else m, channels//dy_reduce if channels//dy_reduce != 0 else 1),
            nn.ReLU(),
            MLP(channels//dy_reduce if channels//dy_reduce != 0 else 1, self.mid_channels) if use_mlp \
                else nn.Linear(channels//dy_reduce if channels//dy_reduce != 0 else 1, self.mid_channels),
            nn.BatchNorm1D(self.mid_channels)
        )

    
    def forward(self, inputs, z_token=None):
        B, C, _, _ = inputs.shape

        if z_token is None:
            x = self.avg_pool(inputs) # B, M, 1, 1
            x = x.reshape(shape=[B, C]) # B, M
            x = self.project(x) # B, 2*k*C
        else:
            B, M, _ = z_token.shape

            x = self.avg_pool(z_token) # B, M, 1, 1
            x = x.reshape(shape=[B, M]) # B, M
            x = self.project(x) # B, 2*k*C

        # 2*self.k: 即获取每个超参数对应的动态参数
        # B, C, 2*self.k
        # 前k个参数为a参数，后k个参数为b参数
        dy_coef = x.reshape(shape=(B, C, 2*self.k)) * \
                    self.lambda_a_b + self.init_value_for_a_b
        x_perm = inputs.transpose(perm=[2, 3, 0, 1]).unsqueeze(-1) # H, W, B, C, 1

        # 每个通道，两种动态结果 == k
        output = x_perm * dy_coef[:, :, :self.k] + dy_coef[:, :, self.k:] # H, W, B, C, self.k
        output = paddle.max(output, axis=-1).transpose(perm=[2, 3, 0, 1]) # B, C, H, W

        return output


class Mobile(nn.Layer):
    """单层的BottleNeck
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 m,
                 t=1,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=DY_ReLU,
                 use_se=False,
                 reduce=4):
        super(Mobile, self).__init__()
        self.m = m
        if stride==2: # keep channels
            self.downsample = DepthWise_Conv(in_channels=in_channels,
                                             out_channels=in_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding)
            self.downsample_bn = nn.BatchNorm2D(in_channels)
            self.downsample_act = act(in_channels*t, m=m, use_mlp=False)
        else:
            self.downsample = None

        self.pointwise_first_block = nn.Sequential(
            PointWise_Conv(in_channels=in_channels,
                           out_channels=in_channels*t),
            nn.BatchNorm2D(in_channels*t)
        )
        self.depthwise_block = nn.Sequential(
            DepthWise_Conv(in_channels=in_channels*t,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding),
            nn.BatchNorm2D(in_channels*t)
        )
        self.pointwise_second_block = nn.Sequential(
            PointWise_Conv(in_channels=in_channels*t,
                           out_channels=out_channels),
            nn.BatchNorm2D(out_channels)
        )

        self.point_act = act(in_channels*t, m=m, use_mlp=False)
        self.depth_act = act(in_channels*t, m=m, use_mlp=False)

        self.bn1 = nn.BatchNorm2D(in_channels*t)
        self.bn2 = nn.BatchNorm2D(in_channels*t)
        self.bn3 = nn.BatchNorm2D(out_channels)
    
    def forward(self, inputs, z_token):
        if self.downsample is None:
            x = self.pointwise_first_block(inputs)
            x = self.point_act(x, z_token=z_token)
        else:
            x = self.downsample(inputs)
            x = self.downsample_bn(x)
            x = self.downsample_act(x, z_token=z_token)
            x = self.pointwise_first_block(x)
            x = self.point_act(x, z_token=z_token)
        
        x = self.depthwise_block(x)
        x = self.depth_act(x, z_token=z_token)

        x = self.pointwise_second_block(x)

        return x


class Attention(nn.Layer):

    def __init__(self,
                 embed_dim,
                 num_head,
                 qkv_bias=True,
                 dropout_rate=0.,
                 attn_dropout_rate=0.):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dims = embed_dim // num_head
        self.scale = self.head_dims ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, 3*self.head_dims*num_head, bias_attr=qkv_bias)

        self.out = nn.Linear(self.head_dims*num_head, embed_dim)

        self.proj_dropout = nn.Dropout(p=dropout_rate)
        self.attn_dropout = nn.Dropout(p=attn_dropout_rate)
        self.softmax = nn.Softmax()
    
    def forward(self, inputs):
        B, N, _ = inputs.shape
        qkv = self.qkv_proj(inputs).chunk(3, axis=-1)
        q, k, v = qkv
        q = q.reshape(shape=[B, N, self.num_head, self.head_dims]).transpose(perm=[0, 2, 1, 3])
        k = k.reshape(shape=[B, N, self.num_head, self.head_dims]).transpose(perm=[0, 2, 1, 3])
        v = v.reshape(shape=[B, N, self.num_head, self.head_dims]).transpose(perm=[0, 2, 1, 3])

        attn = paddle.matmul(q, k, transpose_y=True)
        attn = attn*self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose(perm=[0, 2, 1, 3]).reshape((B, N, self.num_head*self.head_dims))
        z = self.out(z)
        z = self.proj_dropout(z)

        return z

class DropPath(nn.Layer):
    """DropPath class"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        """drop path op
        Args:
            input: tensor with arbitrary shape
            drop_prob: float number of drop path probability, default: 0.0
            training: bool, if current mode is training, default: False
        Returns:
            output: output tensor after drop path
        """
        # if prob is 0 or eval mode, return original input
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = paddle.to_tensor(keep_prob, dtype='float32')
        shape = (inputs.shape[0], ) + (1, ) * (inputs.ndim - 1)  # shape=(N, 1, 1, 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor() # mask
        output = inputs.divide(keep_prob) * random_tensor #divide is to keep same output expectation
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


class Identify(nn.Layer):

    def __init__(self):
        super(Identify, self).__init__()
    
    def forward(self, inputs):
        return inputs


class Former(nn.Layer):
    """全局信息提取
    """
    def __init__(self,
                 d,
                 qkv_bias=False,
                 num_head=1,
                 mlp_ratio=4,
                 dropout_rate=0.,
                 attn_dropout_rate=0.,
                 droppath_rate=0.):
        super(Former, self).__init__()

        self.attn = Attention(d,
                              num_head=num_head,
                              dropout_rate=dropout_rate,
                              attn_dropout_rate=attn_dropout_rate)
        self.attn_bn = nn.LayerNorm(d)

        self.mlp = MLP(in_features=d,
                       hidden_features=int(mlp_ratio*d),
                       drop_rate=dropout_rate)
        self.mlp_bn = nn.LayerNorm(d)

        self.drop_path = DropPath(droppath_rate) if droppath_rate > 0. else Identify()


    def forward(self, inputs):
        res = inputs
        x = self.attn(inputs)
        x = self.attn_bn(x)
        x = self.drop_path(x)
        x = x + res

        res = x
        x = self.mlp(x)
        x = self.mlp_bn(x)
        x = self.drop_path(x)
        x = x + res

        return x

class ToFormer_Bridge(nn.Layer):
    """局部到全局信息桥
    """
    def __init__(self,
                 in_channels,
                 d,
                 num_head=1):
        super(ToFormer_Bridge, self).__init__()
        self.in_channels = in_channels
        self.num_head = num_head
        self.head_dims = in_channels // num_head

        self.k_proj = nn.Linear(in_features=d,
                                out_features=num_head*self.head_dims)
        
        self.v_proj = nn.Linear(in_features=d,
                                out_features=num_head*self.head_dims)

        self.softmax = nn.Softmax()
        

    def forward(self, inputs, z_token):
        B, C, H, W = inputs.shape
        x1 = inputs.reshape([B, C, H*W]).transpose([0, 2, 1]) # B, C, N
        # B, num_head, N, L(in_channels // num_head)
        q = x1.reshape([B, H*W, self.num_head, C//self.num_head]).transpose([0, 2, 1, 3])

        B, M, _ = z_token.shape
        k = self.k_proj(z_token) # B, M, num_head*(in_channels // num_head)
        # B, num_head, M, in_channels // num_head
        k = k.reshape([B, M, self.num_head, self.head_dims]).transpose([0, 2, 1, 3])

        v = self.v_proj(z_token) # B, M, num_head*(in_channels // num_head)
        # B, num_head, M, in_channels // num_head
        v = v.reshape([B, M, self.num_head, self.head_dims]).transpose([0, 2, 1, 3])

        attn = paddle.matmul(q, k, transpose_y=True) # B, num_head, M, N
        attn = self.softmax(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 2, 1, 3]).reshape([B, H*W, self.num_head*self.head_dims])
        z = z.transpose([0, 2, 1]).reshape([B, self.num_head*self.head_dims, H, W])
        z = z + inputs

        return z



class ToMobile_Bridge(nn.Layer):
    """全局到局部信息桥
    """
    def __init__(self,
                 in_channels,
                 d,
                 num_head=1):
        super(ToMobile_Bridge, self).__init__()
        self.in_channels = in_channels
        self.num_head = num_head
        self.head_dims = in_channels // num_head

        self.q_proj = nn.Linear(in_features=d,
                                out_features=num_head*self.head_dims)
        self.out = nn.Linear(in_features=num_head*self.head_dims,
                                out_features=d)

        self.softmax = nn.Softmax()

    def forward(self, inputs, z_token):
        B, C, H, W = inputs.shape
        x1 = inputs.reshape([B, C, H*W]).transpose([0, 2, 1]) # B, C, N
        # B, num_head, N, L(in_channels // num_head)
        x1 = x1.reshape([B, H*W, self.num_head, C//self.num_head]).transpose([0, 2, 1, 3])

        B, M, _ = z_token.shape
        x2 = self.q_proj(z_token) # B, M, num_head*(in_channels // num_head)
        # B, num_head, M, in_channels // num_head
        x2 = x2.reshape([B, M, self.num_head, self.head_dims]).transpose([0, 2, 1, 3])

        attn = paddle.matmul(x2, x1, transpose_y=True) # B, num_head, M, N
        attn = self.softmax(attn)

        z = paddle.matmul(attn, x1) # B, num_head, M, L(in_channels // num_head)
        z = z.transpose([0, 2, 1, 3]).reshape([B, M, self.num_head*self.head_dims])
        z = self.out(z) # B, M, D
        z = z + z_token

        return z


class CombineBlock(nn.Layer):
    """结合Mobile + Bridges + Former
    """
    def __init__(self):
        super(CombineBlock, self).__init__()
        pass

    def forward(self, inputs):
        pass


class MobileFormer(nn.Layer):

    def __init__(self,
                 num_classes=1000,
                 in_channels=3,
                 m=2,
                 d=12):
        super(MobileFormer, self).__init__()
        
        self.M = m
        self.D = d
        self.former_token = self.create_parameter(
            shape=[1, self.M, self.D], dtype='float32',
            attr=nn.initializer.KaimingUniform()
        )


    def forward(self, inputs):
        pass


if __name__ == "__main__":
    data = paddle.empty((1, 3, 32, 32))
    z_token = paddle.empty((1, 2, 8))
    model = ToMobile_Bridge(3, 8, 1)

    y = model(data, z_token)

    print(y.shape)
