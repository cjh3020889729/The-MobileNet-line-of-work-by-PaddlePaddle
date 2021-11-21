import paddle
from paddle import nn
from paddle.nn.layer.activation import GELU
        

class Stem(nn.Layer):
    """渐入层
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.Hardswish):
        super(Stem, self).__init__(
                 name_scope="Stem")
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2D(out_channels)
        self.act = act()
    
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x


class DepthWiseConv(nn.Layer):
    """深度卷积 -- 支持lite形式
    """
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 is_lite=False):
        super(DepthWiseConv, self).__init__(
                 name_scope="DepthWiseConv")
        self.is_lite = is_lite
        if is_lite is False:
            self.conv = nn.Conv2D(in_channels=in_channels,
                                out_channels=in_channels,
                                groups=in_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)
        else:
            self.conv = nn.Sequential(
                # [[0, 1, 2]] -- [3, 1]
                nn.Conv2D(in_channels=in_channels,
                          out_channels=in_channels,
                          kernel_size=[kernel_size, 1],
                          stride=[stride, 1],
                          padding=[padding, 0],
                          groups=in_channels),
                nn.BatchNorm2D(in_channels),
                # [[0], [1], [2]] -- [1, 3]
                nn.Conv2D(in_channels=in_channels,
                          out_channels=in_channels,
                          kernel_size=[1, kernel_size],
                          stride=[1, stride],
                          padding=[0, padding],
                          groups=in_channels)
            )
    
    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class PointWiseConv(nn.Layer):
    """1x1逐点卷积 -- 支持分组卷积
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=1):
        super(PointWiseConv, self).__init__(
                 name_scope="PointWiseConv")
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              groups=groups)
    
    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class DyReLU(nn.Layer):
    """动态激活函数
    """
    def __init__(self,
                 in_channels,
                 embed_dims,
                 k=2, # a_1, a_2 coef, b_1, b_2 bias
                 coefs=[1.0, 0.5], # coef init value
                 consts=[1.0, 0.0], # const init value
                 reduce=4):
        super(DyReLU, self).__init__(
                 name_scope="DyReLU")
        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.k = k

        self.mid_channels = 2*k*in_channels

        # 4 values
        # a_k = alpha_k + coef_k*x, 2
        # b_k = belta_k + coef_k*x, 2
        self.coef = paddle.to_tensor([coefs[0]]*k + [coefs[1]]*k)
        self.const = paddle.to_tensor([consts[0]] + [consts[1]]*(2*k-1))

        self.project = nn.Sequential(
            MLP(in_features=embed_dims,
                out_features=self.mid_channels,
                mlp_ratio=1/reduce,
                act=nn.ReLU),
            nn.BatchNorm(self.mid_channels)
        )
    
    def forward(self, feature_map, tokens):
        B, M, D = tokens.shape
        dy_params = self.project(tokens[:, 0]) # B, mid_channels
        # B, IN_CHANNELS, 2*k
        dy_params = dy_params.reshape(shape=[B, self.in_channels, 2*self.k])

        # B, IN_CHANNELS, 2*k -- a_1, a_2, b_1, b_2
        dy_init_params = dy_params * self.coef + self.const
        f = feature_map.transpose(perm=[2, 3, 0, 1]).unsqueeze(axis=-1) # H, W, B, C, 1

        # output shape: H, W, B, C, k
        output = f * dy_init_params[:, :, :self.k] + dy_init_params[:, :, self.k:]
        output = paddle.max(output, axis=-1) # H, W, B, C
        output = output.transpose(perm=[2, 3, 0, 1]) # B, C, H, W

        return output


class BottleNeck(nn.Layer):
    """瓶颈块
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 groups=1,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 embed_dims=None,
                 k=2, # the number of dyrelu-params
                 coefs=[1.0, 0.5],
                 consts=[1.0, 0.0],
                 reduce=4,
                 use_dyrelu=False,
                 is_lite=False):
        super(BottleNeck, self).__init__(
                 name_scope="BottleNeck")
        self.is_lite = is_lite
        self.use_dyrelu = use_dyrelu

        assert use_dyrelu==False or (use_dyrelu==True and embed_dims is not None), \
               "Error: Please make sure while the use_dyrelu==True,"+\
               " embed_dims(now:{0})>0.".format(embed_dims)


        self.in_pw = PointWiseConv(in_channels=in_channels,
                                   out_channels=hidden_channels,
                                   groups=groups)
        self.in_pw_bn = nn.BatchNorm2D(hidden_channels)
        
        self.dw = DepthWiseConv(in_channels=hidden_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                is_lite=is_lite)
        self.dw_bn = nn.BatchNorm2D(hidden_channels)

        self.out_pw = PointWiseConv(in_channels=hidden_channels,
                                    out_channels=out_channels,
                                    groups=groups)
        self.out_pw_bn = nn.BatchNorm2D(out_channels)

        if use_dyrelu == False:
            self.act = nn.ReLU()
        else:
            self.act = DyReLU(in_channels=hidden_channels,
                                embed_dims=embed_dims,
                                k=k,
                                coefs=coefs,
                                consts=consts,
                                reduce=reduce)

    
    def forward(self, feature_map, tokens):
        x = self.in_pw(feature_map)
        x = self.in_pw_bn(x)
        if self.use_dyrelu:
            x = self.act(x, tokens)

        x = self.dw(x)
        x = self.dw_bn(x)
        if self.use_dyrelu:
            x = self.act(x, tokens)

        x = self.out_pw(x)
        x = self.out_pw_bn(x)

        return x


class Classifier_Head(nn.Layer):
    """分类头
    """
    def __init__(self,
                 in_channels,
                 embed_dims,
                 hidden_features,
                 num_classes=1000,
                 act=nn.Hardswish):
        super(Classifier_Head, self).__init__(
                 name_scope="Classifier_Head")
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=in_channels+embed_dims,
                             out_features=hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features,
                             out_features=num_classes)
        
        self.act = act()
        self.softmax = nn.Softmax()
    
    def forward(self, feature_map, tokens):
        x = self.avg_pool(feature_map) # B, C, 1, 1
        x = self.flatten(x) # B, C
        
        z = tokens[:, 0] # B, 1, D
        x = paddle.concat([x, z], axis=-1)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


class Mobile(nn.Layer):
    """Mobile sub-block
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 embed_dims=None,
                 k=2,
                 coefs=[1.0, 0.5],
                 consts=[1.0, 0.0],
                 reduce=4,
                 use_dyrelu=False):
        super(Mobile, self).__init__(
                 name_scope="Mobile")
        self.add_dw = True if stride==2 else False

        self.bneck = BottleNeck(in_channels=in_channels,
                                hidden_channels=hidden_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=1,
                                groups=groups,
                                embed_dims=embed_dims,
                                k=k,
                                coefs=coefs,
                                consts=consts,
                                reduce=reduce,
                                use_dyrelu=use_dyrelu)

        if self.add_dw: # stride==2
            self.downsample_dw = nn.Sequential(
                DepthWiseConv(in_channels=in_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding),
                nn.BatchNorm2D(in_channels)
                #, nn.ReLU()
            )
    
    def forward(self, feature_map, tokens):
        if self.add_dw:
            feature_map = self.downsample_dw(feature_map)

        x = self.bneck(feature_map, tokens)
        return x


class ToFormer_Bridge(nn.Layer):
    """Mobile to Former Bridge
    """
    def __init__(self,
                 embed_dims,
                 in_channels,
                 num_head=1,
                 dropout_rate=0.,
                 attn_dropout_rate=0.):
        super(ToFormer_Bridge, self).__init__(
                 name_scope="ToFormer_Bridge")
        self.num_head = num_head
        self.head_dims = in_channels // num_head
        self.scale = self.head_dims ** -0.5

        # split head to project
        self.heads_q_proj = []
        for i in range(num_head): # n linear
            self.heads_q_proj.append(
                nn.Linear(in_features=embed_dims // num_head,
                          out_features=self.head_dims)
            )
        self.heads_q_proj = nn.LayerList(self.heads_q_proj)

        self.output = nn.Linear(in_features=self.num_head*self.head_dims,
                                out_features=embed_dims)
        
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_dropout= nn.Dropout(attn_dropout_rate)


    def transfer_shape(self, feature_map, tokens):
        B, C, H, W = feature_map.shape
        assert C % self.num_head == 0, \
            "Erorr: Please make sure feature_map.channels % "+\
            "num_head == 0(now:{0}).".format(C % self.num_head)
        fm = feature_map.reshape(shape=[B, C, H*W]) # B, C, L
        fm = fm.transpose(perm=[0, 2, 1]) # B, L, C -- C = num_head * head_dims
        fm = fm.reshape(shape=[B, H*W, self.num_head, self.head_dims])
        fm = fm.transpose(perm=[0, 2, 1, 3]) # B, n_h, L, h_d

        B, M, D = tokens.shape
        h_token = tokens.reshape(shape=[B, M, self.num_head, D // self.num_head])
        h_token = h_token.transpose(perm=[0, 2, 1, 3]) # B, n_h, M, D // n_h

        return fm, h_token

    def forward(self, feature_map, tokens):
        B, M, D = tokens.shape
        # fm（key/value） to shape: B, n_h, L, h_d
        # token to shape: B, n_h, M, D // n_h
        fm, token = self.transfer_shape(feature_map, tokens)

        q_list = []
        for i in range(self.num_head):
            q_list.append(
                # B, 1, M, head_dims
                self.heads_q_proj[i](token[:, i, :, :]).reshape(
                                shape=[B, 1, M, self.head_dims])
            )
        q = paddle.concat(q_list, axis=1) # B, num_head, M, head_dims

        # attention distribution
        attn = paddle.matmul(q, fm, transpose_y=True) # B, n_h, M, L
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # attention result
        z = paddle.matmul(attn, fm) # B, n_h, M, h_d
        z = z.transpose(perm=[0, 2, 1, 3])
        z = z.reshape(shape=[B, M, self.num_head*self.head_dims])
        z = self.output(z) # B, M, D
        z = self.dropout(z)
        z = z + tokens

        return z


class ToMobile_Bridge(nn.Layer):
    """Former to Mobile Bridge
    """
    def __init__(self,
                 embed_dims,
                 in_channels,
                 num_head=1,
                 dropout_rate=0.,
                 attn_dropout_rate=0.):
        super(ToMobile_Bridge, self).__init__(
                 name_scope="ToMobile_Bridge")
        self.num_head = num_head
        self.head_dims = in_channels // num_head
        self.scale = self.head_dims ** -0.5

        self.heads_k_proj = []
        self.heads_v_proj = []
        for i in range(num_head): # n linear
            self.heads_k_proj.append(
                nn.Linear(in_features=embed_dims // num_head,
                          out_features=self.head_dims)
            )
            self.heads_v_proj.append(
                nn.Linear(in_features=embed_dims // num_head,
                          out_features=self.head_dims)
            )
        self.heads_k_proj = nn.LayerList(self.heads_k_proj)
        self.heads_v_proj = nn.LayerList(self.heads_v_proj)

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_dropout= nn.Dropout(attn_dropout_rate)
    

    def transfer_shape(self, feature_map, tokens):
        B, C, H, W = feature_map.shape
        assert C % self.num_head == 0, \
            "Erorr: Please make sure feature_map.channels % "+\
            "num_head == 0(now:{0}).".format(C % self.num_head)
        fm = feature_map.reshape(shape=[B, C, H*W]) # B, C, L
        fm = fm.transpose(perm=[0, 2, 1]) # B, L, C -- C = num_head * head_dims
        fm = fm.reshape(shape=[B, H*W, self.num_head, self.head_dims])
        fm = fm.transpose(perm=[0, 2, 1, 3]) # B, n_h, L, h_d

        B, M, D = tokens.shape
        k = tokens.reshape(shape=[B, M, self.num_head, D // self.num_head])
        k = k.transpose(perm=[0, 2, 1, 3]) # B, n_h, M, D // n_h
        v = tokens.reshape(shape=[B, M, self.num_head, D // self.num_head])
        v = v.transpose(perm=[0, 2, 1, 3]) # B, n_h, M, D // n_h

        return fm, k, v


    def forward(self, feature_map, tokens):
        B, C, H, W = feature_map.shape
        B, M, D = tokens.shape

        # fm（q） to shape: B, n_h, L, h_d
        # k/v to shape: B, n_h, M, D // n_h
        q, k_, v_ = self.transfer_shape(feature_map, tokens)

        k_list = []
        v_list = []
        for i in range(self.num_head):
            k_list.append(
                # B, 1, M, head_dims
                self.heads_k_proj[i](k_[:, i, :, :]).reshape(
                                shape=[B, 1, M, self.head_dims])
            )
            v_list.append(
                # B, 1, M, head_dims
                self.heads_v_proj[i](v_[:, i, :, :]).reshape(
                                shape=[B, 1, M, self.head_dims])
            )
        k = paddle.concat(k_list, axis=1) # B, num_head, M, head_dims
        v = paddle.concat(v_list, axis=1) # B, num_head, M, head_dims

        # attention distribution
        attn = paddle.matmul(q, k, transpose_y=True) # B, n_h, L, M
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # attention result
        z = paddle.matmul(attn, v) # B, n_h, L, h_d
        z = z.transpose(perm=[0, 1, 3, 2]) # B, n_h, h_d, L
        # B, n_h*h_d, H, W
        z = z.reshape(shape=[B, self.num_head*self.head_dims, H, W])
        z = self.dropout(z)
        z = z + feature_map

        return z


class MLP(nn.Layer):
    """多层感知机
    """
    def __init__(self,
                 in_features,
                 out_features=None,
                 mlp_ratio=2,
                 mlp_dropout_rate=0.,
                 act=nn.GELU):
        super(MLP, self).__init__(name_scope="MLP")
        self.out_features = in_features if out_features is None else \
                            out_features
        
        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=int(mlp_ratio*in_features))
        self.fc2 = nn.Linear(in_features=int(mlp_ratio*in_features),
                             out_features=self.out_features)
        
        self.act = act()
        self.dropout = nn.Dropout(mlp_dropout_rate)
    
    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    """多头注意力机制的实现
    """
    def __init__(self,
                 embed_dims,
                 num_head=1,
                 dropout_rate=0.,
                 attn_dropout_rate=0.):
        super(Attention, self).__init__(
                 name_scope="Attention")
        self.num_head = num_head
        self.head_dims = embed_dims // num_head
        self.scale = self.head_dims ** -0.5

        self.qkv_proj = nn.Linear(in_features=embed_dims,
                                out_features=3*self.num_head*self.head_dims)
        self.output = nn.Linear(in_features=self.num_head*self.head_dims,
                                out_features=embed_dims)
        
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_dropout= nn.Dropout(attn_dropout_rate)
    

    def transfer_shape(self, q, k, v):
        B, M, _ = q.shape
        q = q.reshape(shape=[B, M, self.num_head, self.head_dims])
        q = q.transpose(perm=[0, 2, 1, 3]) # B, n_h, M, h_d
        k = k.reshape(shape=[B, M, self.num_head, self.head_dims])
        k = k.transpose(perm=[0, 2, 1, 3]) # B, n_h, M, h_d
        v = v.reshape(shape=[B, M, self.num_head, self.head_dims])
        v = v.transpose(perm=[0, 2, 1, 3]) # B, n_h, M, h_d

        return q, k, v


    def forward(self, inputs):
        B, M, D = inputs.shape
        assert D % self.num_head == 0, \
            "Erorr: Please make sure Token.D % "+\
            "num_head == 0(now:{0}).".format(D % self.num_head)

        qkv= self.qkv_proj(inputs)
        q, k, v = qkv.chunk(3, axis=-1)
        # B, n_h, M, h_d
        q, k, v = self.transfer_shape(q, k, v)

        attn = paddle.matmul(q, k, transpose_y=True) # B, n_h, M, M
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v) # B, n_h, M, h_d
        z = z.transpose(perm=[0, 2, 1, 3]) # B, M, n_h, h_d
        z = z.reshape(shape=[B, M, self.num_head*self.head_dims])
        z = self.output(z)
        z = self.attn_dropout(z)
        z = z + inputs

        return z


class DropPath(nn.Layer):
    """多分支丢弃层 -- 沿着Batch
    """
    def __init__(self,
                 p=0.):
        super(DropPath, self).__init__(
                 name_scope="DropPath")
        self.p = p
    
    def forward(self, inputs):
        if self.p > 0. and self.training:
            keep_p = 1 - self.p
            keep_p = paddle.to_tensor([keep_p])
            # B, 1, 1....
            shape = [inputs.shape[0]] + [1] * (inputs.ndim-1)
            random_dr = keep_p + paddle.rand(shape=[shape], dtype='float32')
            random_sample = random_dr.floor() # floor to int--B
            output = inputs.divide(keep_p) * random_sample

        return inputs


class Former(nn.Layer):
    """Former
    """
    def __init__(self,
                 embed_dims,
                 num_head=1,
                 mlp_ratio=2,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 attn_dropout_rate=0.,
                 mlp_dropout_rate=0.,
                 norm=nn.LayerNorm,
                 act=nn.GELU):
        super(Former, self).__init__(name_scope="Former")

        self.attn = Attention(embed_dims=embed_dims,
                                num_head=num_head,
                                dropout_rate=dropout_rate,
                                attn_dropout_rate=attn_dropout_rate)
        self.attn_ln = norm(embed_dims)
        self.attn_droppath = DropPath(droppath_rate)

        self.mlp = MLP(in_features=embed_dims,
                        mlp_ratio=mlp_ratio,
                        mlp_dropout_rate=mlp_dropout_rate,
                        act=act)
        self.mlp_ln = norm(embed_dims)
        self.mlp_droppath = DropPath(droppath_rate)
    

    def forward(self, inputs):
        res = inputs
        x = self.attn(inputs)
        x = self.attn_ln(x)
        x = self.attn_droppath(x)
        x = x + res

        res = x
        x = self.mlp(x)
        x = self.mlp_ln(x)
        x = self.mlp_droppath(x)
        x = x + res

        return x


class MFBlock(nn.Layer):
    """MobileFormer的基本组成单元
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 k=2,
                 coefs=[1.0, 0.5],
                 consts=[1.0, 0.0],
                 reduce=4,
                 use_dyrelu=False,
                 num_head=1,
                 mlp_ratio=2,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 attn_dropout_rate=0.,
                 mlp_dropout_rate=0.,
                 norm=nn.LayerNorm,
                 act=nn.GELU):
        super(MFBlock, self).__init__(
                 name_scope="MFBlock")
        self.mobile = Mobile(in_channels=in_channels,
                             hidden_channels=hidden_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             groups=groups,
                             embed_dims=embed_dims,
                             k=k,
                             coefs=coefs,
                             consts=consts,
                             reduce=reduce,
                             use_dyrelu=use_dyrelu)
        
        self.toformer_bridge = ToFormer_Bridge(embed_dims=embed_dims,
                                               in_channels=in_channels,
                                               num_head=num_head,
                                               dropout_rate=dropout_rate,
                                               attn_dropout_rate=attn_dropout_rate)
        
        self.former = Former(embed_dims=embed_dims,
                             num_head=num_head,
                             mlp_ratio=mlp_ratio,
                             dropout_rate=droppath_rate,
                             mlp_dropout_rate=mlp_dropout_rate,
                             attn_dropout_rate=attn_dropout_rate,
                             droppath_rate=droppath_rate,
                             norm=norm,
                             act=act)

        self.tomobile_bridge = ToMobile_Bridge(in_channels=out_channels,
                                               embed_dims=embed_dims,
                                               num_head=num_head,
                                               dropout_rate=dropout_rate,
                                               attn_dropout_rate=attn_dropout_rate)
    
    def forward(self, feature_map, tokens):
        z_h = self.toformer_bridge(feature_map, tokens)
        z_out = self.former(z_h)

        f_h = self.mobile(feature_map, z_out)
        f_out = self.tomobile_bridge(f_h, z_out)

        return f_out, z_out


class MobileFormer(nn.Layer):
    """MobileFormer
    """
    def __init__(self,
                 num_classes=1000,
                 in_channels=3,
                 model_type='26m',
                 mlp_ratio=2,
                 k=2,
                 coefs=[1.0, 0.5],
                 consts=[1.0, 0.0],
                 use_dyrelu=True,
                 dropout_rate=0,
                 droppath_rate=0,
                 attn_dropout_rate=0,
                 mlp_dropout_rate=0,
                 norm=nn.LayerNorm,
                 act=nn.GELU,
                 alpha=1.0):
        super(MobileFormer, self).__init__()
        from configs import update_config
        self.net_config, self.token_config,\
        reduce, num_head, groups = update_config(in_channels=in_channels,
                                            model_type=model_type)
        self.reduce = reduce
        self.num_head = num_head
        self.groups = groups
        # this create a learnable paramater
        self.create_token(self.token_config[0], self.token_config[1])
        embed_dims = self.token_config[1]

        self.stem = Stem(in_channels=self.net_config[0][0],
                         out_channels=int(alpha * self.net_config[0][1]),
                         kernel_size=self.net_config[0][2],
                         stride=self.net_config[0][3],
                         padding=self.net_config[0][4])
        self.bneck_lite = BottleNeck(in_channels=int(alpha * self.net_config[1][0]),
                                     hidden_channels=int(alpha * self.net_config[1][1]),
                                     out_channels=int(alpha * self.net_config[1][2]),
                                     groups=groups,
                                     kernel_size=self.net_config[1][3],
                                     stride=self.net_config[1][4],
                                     padding=self.net_config[1][5],
                                     use_dyrelu=False,
                                     is_lite=True)
    
        self.blocks = []
        for i in range(2, len(self.net_config)-2):
            self.blocks.append(
                MFBlock(
                    in_channels=int(alpha * self.net_config[i][0]),
                    hidden_channels=int(alpha * self.net_config[i][1]),
                    out_channels=int(alpha * self.net_config[i][2]),
                    embed_dims=embed_dims,
                    kernel_size=self.net_config[i][3],
                    stride=self.net_config[i][4],
                    padding=self.net_config[i][5],
                    groups=groups,
                    k=k,
                    coefs=coefs,
                    consts=consts,
                    reduce=reduce,
                    use_dyrelu=use_dyrelu,
                    num_head=num_head,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                    droppath_rate=droppath_rate,
                    attn_dropout_rate=attn_dropout_rate,
                    mlp_dropout_rate=mlp_dropout_rate,
                    norm=norm,
                    act=act
                )
            )
        self.blocks = nn.LayerList(self.blocks)
        
        self.pw = nn.Sequential(
            PointWiseConv(in_channels=int(alpha * self.net_config[-2][0]),
                          out_channels=self.net_config[-2][1],
                          groups=groups),
            nn.BatchNorm2D(self.net_config[-2][1]),
            nn.ReLU()
        )

        self.head = Classifier_Head(in_channels=self.net_config[-1][0],
                                    embed_dims=embed_dims,
                                    hidden_features=self.net_config[-1][1],
                                    num_classes=num_classes)


    def create_token(self, token_size, embed_dims):
        # B(1), token_size, embed_dims
        shape = [1, token_size, embed_dims]
        self.tokens = self.create_parameter(shape=shape, dtype='float32')
    
    def to_batch_tokens(self, batch_size):
        # B, token_size, embed_dims
        return paddle.concat([self.tokens]*batch_size,  axis=0)

    def forward(self, inputs):
        B, _, _, _ = inputs.shape
        # create batch tokens
        tokens = self.to_batch_tokens(B) # B, token_size, embed_dims

        f = self.stem(inputs)
        f = self.bneck_lite(f, tokens)

        for b in self.blocks:
            f, tokens = b(f, tokens)
        
        f = self.pw(f)

        output = self.head(f, tokens)

        return output


def check_model_size(model_class, model_type):
    model = model_class(model_type=model_type)
    paddle.save(model.state_dict(), '_' + model_type + '.pdparams')

    model = paddle.Model(model)
    model_size = model.summary(input_size=(1, 3, 224, 224))['total_params'] / 1000000
    print('----Model [{0}] Size Compare(with paper)----'.format(model_type))
    if model_type == '26m':
        print('(Compare to paper )The now model is more than {0:.4f}M.'.format(model_size - 3.2))
        return
    elif model_type == '52m':
        print('The now model is more than {0:.4f}M.'.format(model_size - 3.5))
        return
    elif model_type == '96m':
        print('The now model is more than {0:.4f}M.'.format(model_size - 4.6))
        return
    elif model_type == '151m':
        print('The now model is more than {0:.4f}M.'.format(model_size - 7.6))
        return
    elif model_type == '214m':
        print('The now model is more than {0:.4f}M.'.format(model_size - 9.4))
        return
    elif model_type == '294m':
        print('The now model is more than {0:.4f}M.'.format(model_size - 11.4))
        return

    print('The now model is more than {0:.4f}M.'.format(model_size - 14.0))




if __name__ == "__main__":
    model_type = '26m'
    check_model_size(MobileFormer, model_type)
    """Model Compare
        Name    Params   Review_Size    Save_Model_Size(.pdparams)
        508M:   14.0M       13.982M          54.724M
        294M:   11.4M       11.390M          44.601M
        214M:   9.4M        9.411M           36.843M
        151M:   7.6M        7.612M           29.814M
        96M:    4.6M        4.602M           18.036M
        52M:    3.5M        3.502M           13.737M
        26M:    3.2M        3.207M           12.586M
    """
