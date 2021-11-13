import paddle
from paddle import nn
from paddle.fluid.dygraph.nn import LayerNorm, Linear
from paddle.fluid.layers.nn import shape
from paddle.framework import dtype
from paddle.nn import functional as F
from paddle.nn.layer import transformer
from paddle.nn.layer.norm import BatchNorm2D

class Identify(nn.Layer):
    """占位符 -- x = f(x)
    """
    def __init__(self):
        super(Identify, self).__init__()

    def forward(self, inputs):
        x = inputs
        return x


class DepthWise_Conv(nn.Layer):
    """深度卷积 -- groups == in_channels
        Params Info:
            in_channels: 输入通道数
            kernel_size: 卷积核大小
            stride: 步长大小
            padding: 填充大小
        Forward Tips:
            - 输入通道数等于输出通道数
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
            "Error: Please make sure the data of input has"+\
            "the same number of channel(in:{0}) with Conv Filter".format(inputs.shape[1])+\
            "Config(set:{0})  in DepthWise_Conv.".format(self.in_channels)
        
        x = self.conv(inputs)

        return x


class PointWise_Conv(nn.Layer):
    """逐点卷积 -- 1x1
        Params Info:
            in_channels: 输入通道数
            out_channels: 输出通道数
        Forward Tips:
            - 输入特征图等于输出特征图
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(PointWise_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0)
    
    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Error: Please make sure the data of input has"+\
            "the same number of channel(in:{0}) with Conv Filter".format(inputs.shape[1])+\
            "Config(set:{0})  in PointWise_Conv.".format(self.in_channels)
        
        x = self.conv(inputs)

        return x

class MLP(nn.Layer):
    """多层感知机 -- Two Linear Layers
        Params Info:
            in_features: 输入特征数
            hidden_features: 隐藏层特征数
            out_features: 输出特征数
            act: 激活函数 -- nn.Layer or nn.functional
            dropout_rate: 丢弃率
        Forward Tips:
            - 输入特征数等于输出特征数
            - 最后输出为线性输出(未act激活)
    """
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features=None,
                 act=nn.GELU,
                 dropout_rate=0.):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = in_features if out_features is None else out_features
        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features,
                             out_features=self.out_features)
        self.act = act()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, inputs):
        assert inputs.shape[-1] == self.in_features, \
            "Error: Please make sure the data of input has"+\
            "the same number of features(in:{0}) with MLP Dense".format(inputs.shape[-1])+\
            "Config(set:{0})  in MLP.".format(self.in_features)
        
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    """注意力模块 -- 支持多头注意力，可控制qkv是否使用bias
        Params Info:
            embed_dims: 输入特征数/嵌入维度
            num_head: 注意力头数
            qkv_bias: qkv映射层是否使用bias, default: True
            dropout_rate: 注意力结果丢弃率
            attn_dropout_rate: 注意力分布丢弃率
        Forward Tips:
            - 输入特征数等于输出特征数
    """
    def __init__(self,
                 embed_dims,
                 num_head=1,
                 qkv_bias=True,
                 dropout_rate=0.,
                 attn_dropout_rate=0.):
        super(Attention, self).__init__()
        self.embed_dims = embed_dims
        self.num_head = num_head
        self.head_dims = embed_dims // num_head
        self.scale = self.head_dims ** -0.5
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate

        self.qkv_proj = nn.Linear(in_features=embed_dims,
                                  out_features=3*self.num_head*self.head_dims,
                                  bias_attr=qkv_bias)
        self.out = nn.Linear(in_features=self.num_head*self.head_dims,
                             out_features=self.embed_dims)
        
        self.softmax = nn.Softmax()
        self.attn_dropout = nn.Dropout(attn_dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)
    
    def transpose_qkv(self, qkv):
        assert len(qkv) == 3, \
            "Error: Please make sure the qkv_params has 3 items"+\
            ", but now it has {0} items in Attention-func:transpose_qkv.".format(len(qkv))

        q, k, v = qkv
        B, M, _ = q.shape
        q = q.reshape(shape=[B, M, self.num_head, self.head_dims]).transpose(perm=[0, 2, 1, 3])
        k = k.reshape(shape=[B, M, self.num_head, self.head_dims]).transpose(perm=[0, 2, 1, 3])
        v = v.reshape(shape=[B, M, self.num_head, self.head_dims]).transpose(perm=[0, 2, 1, 3])
        return q, k, v

    def forward(self, inputs):
        assert inputs.shape[-1] == self.embed_dims, \
            "Error: Please make sure the data of input has"+\
            "the same number of embed_dims(in:{0}) with Attention ".format(inputs.shape[-1])+\
            "Config(set:{0})  in Attention.".format(self.embed_dims)
        
        B, M, D = inputs.shape
        x = self.qkv_proj(inputs) # B, M, 3*self.num_head*self.head_dims
        qkv = x.chunk(3, axis=-1) # [[B, M, self.num_head*self.head_dims], [...], [...]]
        q, k, v = self.transpose_qkv(qkv) # q/k/v_shape: [B, self.num_head, M, self.head_dims]

        attn = paddle.matmul(q, k, transpose_y=True) # B, self.num_head, M, M
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v) # B, self.num_head, M, self.head_dims
        z = z.transpose(perm=[0, 2, 1, 3]) # B, M, self.num_head, self.head_dims
        z = z.reshape(shape=[B, M, self.num_head*self.head_dims])
        z = self.out(z)
        z = self.out_dropout(z)

        return z


class DropPath(nn.Layer):
    """多分支随机丢弃 -- Batch中不同item分支进行随机丢弃
        Params Info:
            p: 丢弃率
        Forward Tips:
            - 返回执行多分支随机丢弃后的结果
            - 输入数据Shape等于输出数据Shape
    """
    def __init__(self, p=0.):
        super(DropPath, self).__init__()
        assert p >= 0. and p <= 1., \
            "Error: Please make sure the drop_rate is limit at [0., 1.],"+\
            "but now it is {0} in DropPath.".format(p)
        
        self.p = p
    
    def forward(self, inputs):
        if self.p > 0. and self.training:

            keep_p = paddle.to_tensor([1 - self.p], dtype='float32')
            random_tensor = keep_p + paddle.rand([inputs.shape[0]]+\
                                                 [1]*(inputs.ndim-1),
                                                 dtype='float32') # shape: [B, 1, 1, 1...]
            random_tensor = random_tensor.floor() # 二值化，向下取整:[0, 1]
            # 除以保持率，是保证drop过程中输出的总期望保持不变
            output = inputs.divide(keep_p) * random_tensor

            return output
        else:
            return inputs


class Former(nn.Layer):
    """Former实现 -- 一个简单的纯transformer结构
        Params Info:
            embed_dims: 输入特征数/嵌入维度
            mlp_ratio: MLP输入特征到隐藏特征的伸缩比例
            num_head: 注意力头数
            qkv_bias: 映射层是否使用bias, default: True
            dropout_rate: 注意力结果丢弃率
            droppath_rate: DropPath多分支丢弃率
            attn_dropout_rate: 注意力分布丢弃率
            mlp_dropout_rate: MLP丢弃率
            act: 激活函数 -- nn.Layer or nn.functional
            norm: 归一化层 -- nn.LayerNorm
        Forward Tips:
            - 输入数据Shape等于输出数据Shape
            - 利用注意力机制计算全局信息
    """
    def __init__(self,
                 embed_dims,
                 mlp_ratio=2,
                 num_head=1,
                 qkv_bias=True,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 attn_dropout_rate=0.,
                 mlp_dropout_rate=0.,
                 act=nn.GELU,
                 norm=nn.LayerNorm):
        super(Former, self).__init__()
        self.embed_dims = embed_dims
        self.mlp_ratio = mlp_ratio
        self.num_head = num_head
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.droppath_rate = droppath_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.mlp_dropout_rate = mlp_dropout_rate

        self.attn = Attention(embed_dims=embed_dims,
                              num_head=num_head,
                              qkv_bias=qkv_bias,
                              dropout_rate=dropout_rate,
                              attn_dropout_rate=attn_dropout_rate)
        self.mlp = MLP(in_features=embed_dims,
                       hidden_features=int(mlp_ratio*embed_dims),
                       dropout_rate=mlp_dropout_rate,
                       act=act)
        
        self.attn_droppath =  DropPath(p=droppath_rate)
        self.mlp_droppath =  DropPath(p=droppath_rate)

        self.attn_norm = norm(embed_dims)
        self.mlp_norm = norm(embed_dims)

    def forward(self, inputs):
        assert inputs.shape[-1] == self.embed_dims, \
            "Error: Please make sure the data of input has"+\
            "the same number of embed_dims(in:{0}) with Former ".format(inputs.shape[-1])+\
            "Config(set:{0})  in Former.".format(self.embed_dims)

        res = inputs
        x = self.attn(inputs)
        x = self.attn_norm(x)
        x = self.attn_droppath(x)
        x = x + res

        res = x
        x = self.mlp(x)
        x = self.mlp_norm(x)
        x = self.mlp_droppath(x)
        x = x + res

        return x


class ToFormer_Bridge(nn.Layer):
    """ToFormer_Bridge实现 -- Mobile-->Former的结构
        Params Info:
            in_channels: 输入特征图通道数
            embed_dims: 输入特征数/嵌入维度
            num_head: 注意力头数
            q_bias: 映射层是否使用bias, default: True
            dropout_rate: 注意力结果丢弃率
            droppath_rate: DropPath多分支丢弃率
            attn_dropout_rate: 注意力分布丢弃率
            norm: 归一化层 -- nn.LayerNorm
        Forward Tips:
            - 输入特征图直接作为Key与Value的序列数据
            - 输入Token映射为Query的序列数据
            - 输入Token序列数据进行局部信息交互并输出等大的Token序列数据
            - 将特征图中提取的局部信息融合进Token的全局信息中
    """
    def __init__(self,
                 in_channels,
                 embed_dims,
                 num_head=1,
                 q_bias=True,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 attn_dropout_rate=0.,
                 norm=nn.LayerNorm):
        super(ToFormer_Bridge, self).__init__()
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_head = num_head
        self.head_dims = in_channels // num_head
        self.scale = self.head_dims ** -0.5
        self.q_bias = q_bias
        self.dropout_rate = dropout_rate
        self.droppath_rate = droppath_rate
        self.attn_dropout_rate = attn_dropout_rate

        self.q_proj = nn.Linear(in_features=embed_dims,
                                out_features=self.num_head*self.head_dims,
                                bias_attr=q_bias)
        self.out = nn.Linear(in_features=self.num_head*self.head_dims,
                             out_features=embed_dims)

        self.softmax = nn.Softmax()
        self.attn_dropout = nn.Dropout(attn_dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)

        self.out_norm = norm(embed_dims)
        self.out_droppath = DropPath(p=droppath_rate)

    def transpose_q(self, q):
        assert q.ndim == 3, \
            "Error: Please make sure the q has 3 dim,"+\
            " but now it is {0} dim in ToFormer_Bridge-func:transpose_q.".format(q.ndim)
        
        B, M, _ = q.shape
        q = q.reshape(shape=[B, M, self.num_head, self.head_dims]).transpose(perm=[0, 2, 1, 3])
        
        return q # B, num_head, M, head_dims
    
    def transform_feature_map(self, feature_map):
        assert feature_map.ndim == 4, \
            "Error: Please make sure the feature_map has 4 dim,"+\
            " but now it is {0} dim in ToFormer_Bridge-func:"+\
            "transform_feature_map.".format(feature_map.ndim)
        
        B, C, H, W = feature_map.shape
        feature_map = feature_map.reshape(shape=[B, C, H*W]
                        ).transpose(perm=[0, 2, 1]) # B, L, C or B, N, C
        feature_map = feature_map.reshape(shape=[B, H*W, self.num_head, self.head_dims])
        feature_map = feature_map.transpose(perm=[0, 2, 1, 3])

        return feature_map # B, num_head, N, head_dims

    def forward(self, feature_map, z_token):
        assert feature_map.shape[1] == self.in_channels, \
            "Error: Please make sure the data of input has "+\
            "the same number of channels(in:{0}) with ToFormer_Bridge ".format(feature_map.shape[1])+\
            "Config(set:{0})  in ToFormer_Bridge.".format(self.in_channels)
        assert z_token.shape[-1] == self.embed_dims, \
            "Error: Please make sure the data of input has "+\
            "the same number of embed_dims(in:{0}) with ToFormer_Bridge ".format(z_token.shape[-1])+\
            "Config(set:{0})  in ToFormer_Bridge.".format(self.embed_dims)

        B, C, H, W = feature_map.shape
        B, M, D = z_token.shape

        q = self.q_proj(z_token) # B, M, num_head*head_dims
        q = self.transpose_q(q=q) # B, num_head, M, head_dims

        k = self.transform_feature_map(feature_map=feature_map) # B, num_head, N, head_dims
        attn = paddle.matmul(q, k, transpose_y=True) # B, num_head, M, N
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        v = self.transform_feature_map(feature_map=feature_map) # B, num_head, N, head_dims
        z = paddle.matmul(attn, v) # B, num_head, M, head_dims
        z = z.transpose(perm=[0, 2, 1, 3]).reshape(shape=[B, M, self.num_head*self.head_dims])
        z = self.out(z)
        z = self.out_dropout(z)

        # 注意力结果标准处理
        z = self.out_norm(z)
        z = self.out_droppath(z)
        z = z + z_token

        return z


class ToMobile_Bridge(nn.Layer):
    """ToMobile_Bridge实现 -- Former-->Mobile的结构
        Params Info:
            in_channels: 输入特征图通道数
            embed_dims: 输入特征数/嵌入维度
            num_head: 注意力头数
            kv_bias: 映射层是否使用bias, default: True
            dropout_rate: 注意力结果丢弃率
            droppath_rate: DropPath多分支丢弃率
            attn_dropout_rate: 注意力分布丢弃率
            norm: 归一化层 -- nn.LayerNorm
        Forward Tips:
            - 输入特征图直接作为Query的序列数据
            - 输入Token映射为Key与Value的序列数据
            - 输入特征图数据进行全局信息交互并输出等大的特征图数据
            - 将Token中提取的全局信息融合进特征图的局部信息中
    """
    def __init__(self,
                 in_channels,
                 embed_dims,
                 num_head=1,
                 kv_bias=True,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 attn_dropout_rate=0.,
                 norm=nn.LayerNorm):
        super(ToMobile_Bridge, self).__init__()
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_head = num_head
        self.head_dims = in_channels // num_head
        self.scale = self.head_dims ** -0.5
        self.kv_bias = kv_bias
        self.dropout_rate = dropout_rate
        self.droppath_rate = droppath_rate
        self.attn_dropout_rate = attn_dropout_rate

        self.kv_proj = nn.Linear(in_features=embed_dims,
                                 out_features=2*self.num_head*self.head_dims,
                                 bias_attr=kv_bias)
        
        # to keep the number of channels
        # not necessary
        self.keep_information_linear = Identify() if (self.head_dims * self.num_head) !=\
                                       self.in_channels else \
                                       nn.Linear(in_features=self.num_head*self.head_dims,
                                                 out_features=in_channels)

        self.softmax = nn.Softmax()
        self.attn_dropout = nn.Dropout(attn_dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)

        self.out_norm = norm(in_channels)
        self.out_droppath = DropPath(p=droppath_rate)

    def transpose_kv(self, kv):
        assert len(kv) == 2, \
            "Error: Please make sure the kv has 2 item,"+\
            " but now it is {0} item in ToMobile_Bridge-func:".format(len(kv))+\
            "transpose_kv."

        k, v = kv
        B, M, _ = k.shape
        # B, num_head, M, head_dims
        k = k.reshape(shape=[B, M, self.num_head, self.head_dims]).transpose(perm=[0, 2, 1, 3])
        v = v.reshape(shape=[B, M, self.num_head, self.head_dims]).transpose(perm=[0, 2, 1, 3])
        
        return k, v

    def transform_feature_map(self, feature_map):
        assert feature_map.ndim == 4, \
            "Error: Please make sure the feature_map has 4 dim,"+\
            " but now it is {0} dim in ToMobile_Bridge-func:"+\
            "transform_feature_map.".format(feature_map.ndim)

        B, C, H, W = feature_map.shape
        feature_map = feature_map.reshape(shape=[B, C, H*W]
                        ).transpose(perm=[0, 2, 1]) # B, L, C or B, N, C
        feature_map = feature_map.reshape(shape=[B, H*W, self.num_head, self.head_dims])
        feature_map = feature_map.transpose(perm=[0, 2, 1, 3])

        return feature_map # B, num_head, N, head_dims

    def forward(self, feature_map, z_token):
        assert feature_map.shape[1] == self.in_channels, \
            "Error: Please make sure the data of input has "+\
            "the same number of channels(in:{0}) with ToMobile_Bridge ".format(feature_map.shape[1])+\
            "Config(set:{0})  in ToMobile_Bridge.".format(self.in_channels)
        assert z_token.shape[-1] == self.embed_dims, \
            "Error: Please make sure the data of input has "+\
            "the same number of embed_dims(in:{0}) with ToMobile_Bridge ".format(z_token.shape[-1])+\
            "Config(set:{0})  in ToMobile_Bridge.".format(self.embed_dims)

        B, C, H, W = feature_map.shape # N=H*W
        B, M, D = z_token.shape

        x = self.kv_proj(z_token)
        kv = x.chunk(2, axis=-1)
        k, v = self.transpose_kv(kv=kv) # B, num_head, M, head_dims
        q = self.transform_feature_map(feature_map=feature_map) # B, num_head, N, head_dims

        attn = paddle.matmul(q, k, transpose_y=True) # B, num_head, N, M
        attn = attn*self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v) # B, num_head, N, head_dims
        z = z.transpose(perm=[0, 2, 1, 3]) # B, N, num_head, head_dims
        z = z.reshape(shape=[B, H*W, self.num_head*self.head_dims])
        z = self.keep_information_linear(z) # keep the number of channels
        z = self.out_dropout(z)

        # 注意力结果标准处理
        z = self.out_norm(z)
        z = self.out_droppath(z)
        z = z.transpose(perm=[0, 2, 1]).reshape(shape=[B, C, H, W])
        z = z + feature_map

        return z


class DY_ReLU(nn.Layer):
    """DY_ReLU实现 -- Token作为系数输入，特征图作为激活输入
        Params Info:
            in_channels: 输入特征图通道数
            m: 输入Token长度/序列长度
            k: 动态参数基数,
               k=2 --> 每个完整的动态参数集都包含[a_1, a_2] 和[b_1, b_2]，类推 
            use_mlp: 动态参数映射层是否使用MLP, default:False
            coefs: 计算[a_1, a_2] 和[b_1, b_2]时的系数，超参数，default:[1.0, 0.5]
            const: 计算[a_1, a_2] 和[b_1, b_2]时的常数，超参数，default:[1.0, 0.0]
            dy_reduce: 动态参数映射的隐藏映射伸缩比例，基于输入特征图通道数
        Forward Tips:
            - 输入特征图作为激活输入
            - 输入Token作为系数输入
            - 1.Token数据经过动态参数映射得到的结果，参与[a_1, a_2] 和[b_1, b_2]的计算
            - 2.每个Token元素(值)都生成一组[a_1(x), a_2(x)] 和[b_1(x), b_2(x)]
            - 3.将Token生成结果与输入特征图进行交互，沿通道维度作用——每个原始输入生成4个激活可选值
            - 4.经过max选择最大的作为动态ReLU的输出
    """
    def __init__(self,
                 in_channels,
                 m=3,
                 k=2,
                 use_mlp=False,
                 coefs=[1.0, 0.5],
                 const=[1.0, 0.0],
                 dy_reduce=6):
        super(DY_ReLU, self).__init__()
        self.in_channels = in_channels
        self.m = m
        self.k = k
        self.use_mlp = use_mlp
        self.coefs = coefs # [0]:a系数，[1]:b系数
        self.const = const # [0]:a初始常数值，[1]:b初始常数值
        self.dy_reduce = dy_reduce

        # k*in_channels，是为了方便沿通道维度进行动态ReLU计算
        self.mid_channels = 2*k*in_channels # 乘以2，是为了同时构建a，b的参数群

        # 构建每个a/b的系数
        # 因为k=2, 因此a和b参数群各有两个系数
        self.lambda_coefs = paddle.to_tensor([self.coefs[0]]*k+\
                                             [self.coefs[1]]*k,
                                             dtype='float32')
        self.init_consts = paddle.to_tensor([self.const[0]]+\
                                            [self.const[1]]*(2*k-1),
                                            dtype='float32')

        # 这里池化Token数据，所以用一维池化
        self.avg_pool = nn.AdaptiveAvgPool1D(output_size=1)
        self.project = nn.Sequential(
            MLP(in_features=m, out_features=in_channels//dy_reduce) if use_mlp is None else \
            nn.Linear(in_features=m, out_features=in_channels//dy_reduce),
            nn.ReLU(),
            MLP(in_features=in_channels//dy_reduce, out_features=self.mid_channels) if use_mlp is None else \
            nn.Linear(in_features=in_channels//dy_reduce, out_features=self.mid_channels),
            nn.BatchNorm1D(self.mid_channels)
        )
    
    def forward(self, feature_map, z_token):
        assert feature_map.shape[1] == self.in_channels, \
            "Error: Please make sure the feature_map of input has "+\
            "the same number of channels(in:{0}) with DY_ReLU ".format(feature_map.shape[1])+\
            "Config(set:{0})  in DY_ReLU.".format(self.in_channels)
        assert z_token.shape[1] == self.m, \
            "Error: Please make sure the z_token of input has "+\
            "the same number of sequence(in:{0}) with DY_ReLU ".format(z_token.shape[1])+\
            "Config(set:{0})  in DY_ReLU.".format(self.m)

        B, C, H, W = feature_map.shape
        B, M, _ = z_token.shape
        x = self.avg_pool(z_token) # B, M, 1
        x = x.reshape(shape=[B, M]) # B, M
        x = self.project(x) # B, 2*K*C
        x = x.reshape(shape=[B, C, 2*self.k]) # B, C, 2*K

        # 利用超参数计算动态系数
        dy_coef = x * self.lambda_coefs + self.init_consts # B, C, 2*K
        # 其中，2*K包含a参数群与b参数群
        # a为代入系数，b为带入偏置(bias)
        # 动态ReLU，激活值计算函数: f(x)_k = a_k*x + b_k
        # 这里计算通道上的动态ReLU交互
        x_perm = feature_map.transpose(perm=[2, 3, 0, 1]) # H, W, B, C
        # 此时，再添加一个最低维度——对齐: B, C, 2*K, 以保证广播运算
        x_perm = x_perm.unsqueeze(-1) # H, W, B, C, 1

        # 实现: f(x)_k = a_k*x + b_k
        output = x_perm * dy_coef[:, :, :self.k] + dy_coef[:, :, self.k:] # H, W, B, C, K
        # 最后执行激活输出，判断函数为: max(c_f(x)_k)
        # 沿着通道维度向下进行max取值
        output = paddle.max(output, axis=-1) # H, W, B, C
        # 还原输入特征图形式
        output = output.transpose(perm=[2, 3, 0, 1]) # H, W, B, C

        return output


class Mobile(nn.Layer):
    """Mobile实现 -- 特征图作为特征输入，Token作为参数选择输入(DY_ReLU)
        Params Info:
            in_channels: 输入特征图通道数
            out_channels: 输出特征图通道数
            t: Bottle结构中的通道伸缩比例
            m: 输入Token长度/序列长度
            kernel_size: 卷积核大小
            stride: 步长大小
            padding: 填充大小
            k: 动态参数基数,
               k=2 --> 每个完整的动态参数集都包含[a_1, a_2] 和[b_1, b_2]，类推 
            use_mlp: 动态参数映射层是否使用MLP, default:False
            coefs: 计算[a_1, a_2] 和[b_1, b_2]时的系数，超参数，default:[1.0, 0.5]
            const: 计算[a_1, a_2] 和[b_1, b_2]时的常数，超参数，default:[1.0, 0.0]
            dy_reduce: 动态参数映射的隐藏映射伸缩比例，基于输入特征图通道数
        Forward Tips:
            - 输入特征图作为特征输入
            - 输入Token作为参数选择输入(DY_ReLU)
            - 1.如果stride为2，则额外添加一次下采样--利用DepthWise_Conv + BacthNorm2D + DY_ReLU
            - 2.输入特征图进入Bottle结构，顺序经过PW，DY_ReLU，DW，DY_ReLU，PW
            - 3.Token参数进入DY_ReLU中提供动态参数
            - 4.输出特征提取后的特征图
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 t=4,
                 m=3,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 k=2,
                 use_mlp=False,
                 coefs=[1.0, 0.5],
                 const=[1.0, 0.0],
                 dy_reduce=6):
        super(Mobile, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t = t
        self.m = m
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.k = k
        self.use_mlp = use_mlp
        self.coefs = coefs
        self.const = const
        self.dy_reduce = dy_reduce

        if stride == 2:
            self.downsample = nn.Sequential(
                DepthWise_Conv(in_channels=in_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding),
                nn.BatchNorm2D(in_channels)
            )
            self.downsample_act = DY_ReLU(in_channels=in_channels,
                                            m=m, k=k, use_mlp=use_mlp,
                                            coefs=coefs, const=const,
                                            dy_reduce=dy_reduce)
        else:
            self.downsample = Identify()
        
        self.in_pointwise_conv = nn.Sequential(
            PointWise_Conv(in_channels=in_channels,
                           out_channels=int(t * in_channels)),
            nn.BatchNorm2D(int(t * in_channels))
        )
        self.in_pointwise_conv_act = DY_ReLU(in_channels=int(t * in_channels),
                                                m=m, k=k, use_mlp=use_mlp,
                                                coefs=coefs, const=const,
                                                dy_reduce=dy_reduce)

        self.hidden_depthwise_conv = nn.Sequential(
            DepthWise_Conv(in_channels=int(t * in_channels),
                           kernel_size=kernel_size,
                           stride=1,
                           padding=1),
            nn.BatchNorm2D(int(t * in_channels))
        )
        self.hidden_depthwise_conv_act = DY_ReLU(in_channels=int(t * in_channels),
                                                    m=m, k=k, use_mlp=use_mlp,
                                                    coefs=coefs, const=const,
                                                    dy_reduce=dy_reduce)

        self.out_pointwise_conv = nn.Sequential(
            PointWise_Conv(in_channels=int(t * in_channels),
                           out_channels=out_channels),
            nn.BatchNorm2D(out_channels)
        )

    def forward(self, feature_map, z_token):
        assert feature_map.shape[1] == self.in_channels, \
            "Error: Please make sure the feature_map of input has "+\
            "the same number of channels(in:{0}) with Mobile ".format(feature_map.shape[1])+\
            "Config(set:{0})  in Mobile.".format(self.in_channels)
        assert z_token.shape[1] == self.m, \
            "Error: Please make sure the z_token of input has "+\
            "the same number of sequence(in:{0}) with Mobile ".format(z_token.shape[1])+\
            "Config(set:{0})  in Mobile.".format(self.m)

        x = self.downsample(feature_map)
        if self.stride == 2:
            x = self.downsample_act(x, z_token)
        x = self.in_pointwise_conv(x)
        x = self.in_pointwise_conv_act(x, z_token)
        x = self.hidden_depthwise_conv(x)
        x = self.hidden_depthwise_conv_act(x, z_token)
        x = self.out_pointwise_conv(x)

        return x


class Stem(nn.Layer):
    """渐入层 -- 使用V3的Hardswish激活函数
        Params Info:
            in_channels: 输入特征图通道数
            out_channels: 输出特征图通道数
            kernel_size: 卷积核大小
            stride: 步长大小
            padding: 填充大小
            act: 激活函数 -- nn.Layer or nn.functional
        Forward Tips:
            - 可以直接替换激活函数 -- nn.ReLU等
            - 动态ReLU需要内部配置，直接设置act无效
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.Hardswish):
        super(Stem, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2D(out_channels)
        self.act = act()
    
    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Error: Please make sure the data of input has "+\
            "the same number of channels(in:{0}) with Stem ".format(inputs.shape[1])+\
            "Config(set:{0})  in Stem.".format(self.in_channels)

        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)

        return x


class Lite_BottleNeck(nn.Layer):
    """轻量BottleNeck -- 针对DepthWise_Conv进行卷积分解，使用激活函数ReLU6
        Params Info:
            in_channels: 输入特征图通道数
            hidden_channels: 深度卷积中通道渐变后的通道数
            out_channels: 输出特征图通道数
            expands: 深度卷积中通道第一次渐变的伸缩比例，1/expands
            kernel_size: 卷积核大小
            stride: 步长大小
            padding: 填充大小
            act: 激活函数 -- nn.Layer or nn.functional
        Forward Tips:
            - 可以直接替换激活函数 -- nn.ReLU等
            - DepthWise_Conv被分解，因此计算更轻量
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 expands=2,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.ReLU6):
        super(Lite_BottleNeck, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.expands = expands
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        first_filter = [kernel_size, 1]
        first_stride = [stride, 1]
        first_padding = [padding, 0]
        second_filter = [1, kernel_size]
        second_stride = [1, stride]
        second_padding = [0, padding]

        self.lite_depthwise_conv = nn.Sequential(
            nn.Conv2D(in_channels=in_channels,
                      out_channels=hidden_channels - ((hidden_channels - in_channels) // self.expands),
                      kernel_size=first_filter,
                      stride=first_stride,
                      padding=first_padding,
                      groups=self.Gcd(in_channels=in_channels,
                                      hidden_channels=hidden_channels - \
                                      ((hidden_channels - in_channels) // self.expands))),
            nn.BatchNorm2D(hidden_channels - ((hidden_channels - in_channels) // self.expands)),
            nn.Conv2D(in_channels=hidden_channels - ((hidden_channels - in_channels) // self.expands),
                      out_channels=hidden_channels,
                      kernel_size=second_filter,
                      stride=second_stride,
                      padding=second_padding,
                      groups=self.Gcd(hidden_channels=hidden_channels,
                                      in_channels=hidden_channels - \
                                      ((hidden_channels - in_channels) // self.expands))),
            nn.BatchNorm2D(hidden_channels),
            act()
        )

        self.pointwise_conv = nn.Sequential(
            PointWise_Conv(in_channels=hidden_channels,
                           out_channels=out_channels),
            nn.BatchNorm2D(out_channels),
            act()
        )

    def Gcd(self, in_channels, hidden_channels):
        """欧几里得取最大group数
        """
        m = in_channels
        n = hidden_channels
        r = 0
        while n > 0:
            r = m % n
            m = n
            n = r

        return m
    
    def forward(self, inputs):
        assert inputs.shape[1] == self.in_channels, \
            "Error: Please make sure the data of input has"+\
            "the same number of channels(in:{0}) with Lite_BottleNeck ".format(inputs.shape[1])+\
            "Config(set:{0})  in Lite_BottleNeck.".format(self.in_channels)
        
        x = self.lite_depthwise_conv(inputs)
        x = self.pointwise_conv(x)

        return x


def get_head_hidden_size(head_type='base'):
    if head_type == 'base' or head_type == 'big':
        return 1920
    elif head_type == 'base_small':
        return 1600
    elif head_type == 'mid' or head_type == 'mid_small':
        return 1280
    elif head_type == 'samll' or head_type == 'tiny':
        return 1024
    
    return 1920


class Classifier_Head(nn.Layer):
    """Classifier_Head -- 输出分类结果
        Params Info:
            in_channels: 输入特征图通道数
            embed_dims: 输出Token嵌入维度
            expands: 深度卷积中通道渐变的伸缩比例
            num_classes: 分类数
            head_type: 分类头类型，自动调节不同模型的隐藏层大小
            act: 激活函数 -- nn.Layer or nn.functional
        Forward Tips:
            - 可以直接替换激活函数 -- nn.ReLU等
    """
    def __init__(self,
                 in_channels,
                 embed_dims,
                 num_classes,
                 head_type='base',
                 act=nn.Hardswish):
        super(Classifier_Head, self).__init__()
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.head_type = head_type

        self.head_hidden_size = get_head_hidden_size(head_type)

        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc = nn.Linear(in_features=in_channels + embed_dims,
                            out_features=self.head_hidden_size)
        self.out = nn.Linear(in_features=self.head_hidden_size,
                            out_features=num_classes)

        self.flatten = nn.Flatten()
        self.act = act()
        self.softmax = nn.Softmax()
    
    def forward(self, feature_map, z_token):
        assert feature_map.shape[1] == self.in_channels, \
            "Error: Please make sure the feature_map of input has "+\
            "the same number of channels(in:{0}) with Classifier_Head ".format(feature_map.shape[1])+\
            "Config(set:{0})  in Classifier_Head.".format(self.in_channels)
        assert z_token.shape[-1] == self.embed_dims, \
            "Error: Please make sure the z_token of input has "+\
            "the same number of embed_dims(in:{0}) with Classifier_Head ".format(z_token.shape[-1])+\
            "Config(set:{0})  in Classifier_Head.".format(self.embed_dims)

        x = self.avg_pool(feature_map)
        x = self.flatten(x) # B, C

        cls_token = z_token[:, 0] # B, D
        x = paddle.concat([x, cls_token], axis=-1) # B, C+D

        x = self.fc(x)
        x = self.act(x)
        x = self.out(x)
        x = self.softmax(x)

        return x


class Basic_Block(nn.Layer):
    """MobileFormer最小基础模块
        Params Info:
            embed_dims: 
            in_channels: 
            out_channels: 
            num_head: 
            mlp_ratio: 
            q_bias: 
            kv_bias: 
            qkv_bias: 
            dropout_rate: 
            droppath_rate: 
            attn_dropout_rate: 
            mlp_dropout_rate: 
            t: 
            m: 
            kernel_size: 
            stride: 
            padding: 
            k: 
            use_mlp: 
            coefs: 
            const: 
            dy_reduce: 
            add_pointwise_conv: 
            pointwise_conv_channels: 
            act: 
            norm: 
        Forward Tips:
            - ToFormer_Bridge --> Former --> Mobile --> ToMobile_Bridge
    """
    def __init__(self,
                 embed_dims,
                 in_channels,
                 out_channels,
                 num_head=1,
                 mlp_ratio=2,
                 q_bias=True,
                 kv_bias=True,
                 qkv_bias=True,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 attn_dropout_rate=0.,
                 mlp_dropout_rate=0.,
                 t=1,
                 m=6,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 k=2,
                 use_mlp=False,
                 coefs=[1.0, 0.5],
                 const=[1.0, 0.0],
                 dy_reduce=6,
                 add_pointwise_conv=False,
                 pointwise_conv_channels=None,
                 act=nn.GELU,
                 norm=nn.LayerNorm):
        super(Basic_Block, self).__init__()

        self.add_pointwise_conv = add_pointwise_conv
        self.pointwise_conv_channels = pointwise_conv_channels

        self.mobile = Mobile(in_channels=in_channels,
                             out_channels=out_channels,
                             t=t, m=m, kernel_size=kernel_size,
                             stride=stride, padding=padding, k=k,
                             use_mlp=use_mlp, coefs=coefs,
                             const=const, dy_reduce=dy_reduce)

        self.toformer_bridge = ToFormer_Bridge(in_channels=in_channels,
                                               embed_dims=embed_dims,
                                               num_head=num_head,
                                               q_bias=q_bias,
                                               dropout_rate=dropout_rate,
                                               droppath_rate=droppath_rate,
                                               attn_dropout_rate=attn_dropout_rate,
                                               norm=norm)
        
        self.former = Former(embed_dims=embed_dims,
                             mlp_ratio=mlp_ratio,
                             num_head=num_head,
                             qkv_bias=qkv_bias,
                             dropout_rate=dropout_rate,
                             droppath_rate=droppath_rate,
                             attn_dropout_rate=attn_dropout_rate,
                             mlp_dropout_rate=mlp_dropout_rate,
                             act=act,
                             norm=norm)
        
        self.tomobile_bridge = ToMobile_Bridge(in_channels=out_channels,
                                               embed_dims=embed_dims,
                                               num_head=num_head,
                                               kv_bias=kv_bias,
                                               dropout_rate=dropout_rate,
                                               droppath_rate=droppath_rate,
                                               attn_dropout_rate=attn_dropout_rate,
                                               norm=norm)
        
        if add_pointwise_conv == True:
            self.pointwise_conv = PointWise_Conv(in_channels=out_channels,
                                                 out_channels=pointwise_conv_channels)
        else:
            self.pointwise_conv = Identify()
    
    def forward(self, feature_map, z_token):
        assert feature_map.shape[1] == self.in_channels, \
            "Error: Please make sure the feature_map of input has "+\
            "the same number of channels(in:{0}) with Basic_Block ".format(feature_map.shape[1])+\
            "Config(set:{0})  in Basic_Block.".format(self.in_channels)
        assert z_token.shape[-1] == self.embed_dims, \
            "Error: Please make sure the z_token of input has "+\
            "the same number of embed_dims(in:{0}) with Basic_Block ".format(z_token.shape[-1])+\
            "Config(set:{0})  in Basic_Block.".format(self.embed_dims)

        z = self.toformer_bridge(feature_map, z_token)
        z = self.former(z)

        x = self.mobile(feature_map, z)
        output_x = self.tomobile_bridge(x, z)
        output_x = self.pointwise_conv(output_x)

        return output_x, z
