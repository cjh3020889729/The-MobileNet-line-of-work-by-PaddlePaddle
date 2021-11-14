import paddle
from paddle import nn


def MobileFormer_Config_Names():
    names_ = [
                'big', 'base', 'base_small',
                'mid', 'mid_small', 'small',
                'tiny'
             ]
    return names_

def get_head_hidden_size(head_type='base'):
    """Head隐藏层大小
    """
    if head_type == 'base' or head_type == 'big':
        return 1920
    elif head_type == 'base_small':
        return 1600
    elif head_type == 'mid' or head_type == 'mid_small':
        return 1280
    elif head_type == 'small' or head_type == 'tiny':
        return 1024
    
    return 1920

def MobileFormer_Big_Config(alpha=1.0, print_info=False):
    """配置对应原文: Mobile-Former-508M
    """
    net_configs = {
            # Config Params:
            # {'base_cfg':..., 'stem':..., 'bneck-lite':..., 'MF':..., 'head':...}
            # base_cfg: MobileFormer基本配置参数
            # stem: Stem层配置参数
            # bneck-lite: Lite_BottleNeck层配置参数
            # MF: Basic_Block层配置参数
            # head: Classifier_Head层配置参数
            'base_cfg': [ # m, embed_dims, num_head, mlp_ratio,
                          # q_bias, kv_bias, qkv_bias,
                          # dropout_rate, droppath_rate,
                          # attn_dropout_rate, mlp_dropout_rate,
                          # k, use_mlp,
                          # coefs, const, dy_reduce, groups
                            6, 192, 8, 2,
                            True, True, True,
                            0., 0.,
                            0., 0.,
                            2, True,
                            [1.0, 0.5], [1.0, 0.0], 6, 1
                        ],

            'stem': [ # in_channels, out_channels, kernel_size, stride, padding, act
                        3, int(alpha*24), 3, 2, 1, nn.Hardswish
                    ],

            'bneck-lite': [ # in_channels, hidden_channels, out_channels, 
                            # expands, kernel_size, stride, padding, act
                              int(alpha*24), int(alpha*48), int(alpha*24),
                              2, 3, 1, 1, nn.ReLU6
                          ],
            
            'MF': [ # in_channels, out_channels,
                    # t, kernel_size, stride, padding,
                    # add_pointwise_conv, pointwise_conv_channels,
                    # act, norm
                      [
                          int(alpha*24), int(alpha*40),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_1 Config(downsample)
                      [
                          int(alpha*40), int(alpha*40),
                          3, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_2 Config
                      [
                          int(alpha*40), int(alpha*72),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_3 Config(downsample)
                      [
                          int(alpha*72), int(alpha*72),
                          3, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_4 Config
                      [
                          int(alpha*72), int(alpha*128),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_5 Config(downsample)
                      [
                          int(alpha*128), int(alpha*128),
                          4, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_6 Config
                      [
                          int(alpha*128), int(alpha*176),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_7 Config
                      [
                          int(alpha*176), int(alpha*176),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_8 Config
                      [
                          int(alpha*176), int(alpha*240),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_9 Config(downsample)
                      [
                          int(alpha*240), int(alpha*240),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_10 Config
                      [
                          int(alpha*240), int(alpha*240),
                          6, 3, 1, 1,
                          True, 1440,
                          nn.GELU, nn.LayerNorm
                      ] # MF_11 Config
                  ],

            'head': [ # in_channels, head_type, act
                        1440, 'big', nn.Hardswish
                    ]
        }

    if print_info == True:
        parse_config(net_configs)
        print('------------------Config Info--------------------')
        print('The Create Mobile-Former Model Has -{0}- MF-Block'.format(len(net_configs['MF'])))
        print('------------------Config Info--------------------\n')

    return net_configs

def MobileFormer_Base_Config(alpha=1.0, print_info=False):
    """配置对应原文: Mobile-Former-294M
    """
    net_configs = {
            # Config Params:
            # {'base_cfg':..., 'stem':..., 'bneck-lite':..., 'MF':..., 'head':...}
            # base_cfg: MobileFormer基本配置参数
            # stem: Stem层配置参数
            # bneck-lite: Lite_BottleNeck层配置参数
            # MF: Basic_Block层配置参数
            # head: Classifier_Head层配置参数
            'base_cfg': [ # m, embed_dims, num_head, mlp_ratio,
                          # q_bias, kv_bias, qkv_bias,
                          # dropout_rate, droppath_rate,
                          # attn_dropout_rate, mlp_dropout_rate,
                          # k, use_mlp,
                          # coefs, const, dy_reduce, groups
                            6, 192, 8, 2,
                            True, True, True,
                            0., 0.,
                            0., 0.,
                            2, True,
                            [1.0, 0.5], [1.0, 0.0], 6, 1
                        ],

            'stem': [ # in_channels, out_channels, kernel_size, stride, padding, act
                        3, int(alpha*16), 3, 2, 1, nn.Hardswish
                    ],

            'bneck-lite': [ # in_channels, hidden_channels, out_channels, 
                            # expands, kernel_size, stride, padding, act
                              int(alpha*16), int(alpha*32), int(alpha*16),
                              2, 3, 1, 1, nn.ReLU6
                          ],
            
            'MF': [ # in_channels, out_channels,
                    # t, kernel_size, stride, padding,
                    # add_pointwise_conv, pointwise_conv_channels,
                    # act, norm
                      [
                          int(alpha*16), int(alpha*24),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_1 Config(downsample)
                      [
                          int(alpha*24), int(alpha*24),
                          4, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_2 Config
                      [
                          int(alpha*24), int(alpha*48),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_3 Config(downsample)
                      [
                          int(alpha*48), int(alpha*48),
                          4, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_4 Config
                      [
                          int(alpha*48), int(alpha*96),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_5 Config(downsample)
                      [
                          int(alpha*96), int(alpha*96),
                          4, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_6 Config
                      [
                          int(alpha*96), int(alpha*128),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_7 Config
                      [
                          int(alpha*128), int(alpha*128),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_8 Config
                      [
                          int(alpha*128), int(alpha*192),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_9 Config(downsample)
                      [
                          int(alpha*192), int(alpha*192),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_10 Config
                      [
                          int(alpha*192), int(alpha*192),
                          6, 3, 1, 1,
                          True, 1152,
                          nn.GELU, nn.LayerNorm
                      ] # MF_11 Config
                  ],

            'head': [ # in_channels, head_type, act
                        1152, 'base', nn.Hardswish
                    ]
        }

    if print_info == True:
        parse_config(net_configs)
        print('------------------Config Info--------------------')
        print('The Create Mobile-Former Model Has -{0}- MF-Block'.format(len(net_configs['MF'])))
        print('------------------Config Info--------------------\n')

    return net_configs

def MobileFormer_Base_Small_Config(alpha=1.0, print_info=False):
    """配置对应原文: Mobile-Former-214M
    """
    net_configs = {
            # Config Params:
            # {'base_cfg':..., 'stem':..., 'bneck-lite':..., 'MF':..., 'head':...}
            # base_cfg: MobileFormer基本配置参数
            # stem: Stem层配置参数
            # bneck-lite: Lite_BottleNeck层配置参数
            # MF: Basic_Block层配置参数
            # head: Classifier_Head层配置参数
            'base_cfg': [ # m, embed_dims, num_head, mlp_ratio,
                          # q_bias, kv_bias, qkv_bias,
                          # dropout_rate, droppath_rate,
                          # attn_dropout_rate, mlp_dropout_rate,
                          # k, use_mlp,
                          # coefs, const, dy_reduce, groups
                            6, 192, 4, 2,
                            True, True, True,
                            0., 0.,
                            0., 0.,
                            2, True,
                            [1.0, 0.5], [1.0, 0.0], 6, 1
                        ],

            'stem': [ # in_channels, out_channels, kernel_size, stride, padding, act
                        3, int(alpha*12), 3, 2, 1, nn.Hardswish
                    ],

            'bneck-lite': [ # in_channels, hidden_channels, out_channels, 
                            # expands, kernel_size, stride, padding, act
                              int(alpha*12), int(alpha*24), int(alpha*12),
                              2, 3, 1, 1, nn.ReLU6
                          ],
            
            'MF': [ # in_channels, out_channels,
                    # t, kernel_size, stride, padding,
                    # add_pointwise_conv, pointwise_conv_channels,
                    # act, norm
                      [
                          int(alpha*12), int(alpha*20),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_1 Config(downsample)
                      [
                          int(alpha*20), int(alpha*20),
                          3, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_2 Config
                      [
                          int(alpha*20), int(alpha*40),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_3 Config(downsample)
                      [
                          int(alpha*40), int(alpha*40),
                          4, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_4 Config
                      [
                          int(alpha*40), int(alpha*80),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_5 Config(downsample)
                      [
                          int(alpha*80), int(alpha*80),
                          4, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_6 Config
                      [
                          int(alpha*80), int(alpha*112),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_7 Config
                      [
                          int(alpha*112), int(alpha*112),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_8 Config
                      [
                          int(alpha*112), int(alpha*160),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_9 Config(downsample)
                      [
                          int(alpha*160), int(alpha*160),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_10 Config
                      [
                          int(alpha*160), int(alpha*160),
                          6, 3, 1, 1,
                          True, 960,
                          nn.GELU, nn.LayerNorm
                      ] # MF_11 Config
                  ],

            'head': [ # in_channels, head_type, act
                        960, 'base_small', nn.Hardswish
                    ]
        }

    if print_info == True:
        parse_config(net_configs)
        print('------------------Config Info--------------------')
        print('The Create Mobile-Former Model Has -{0}- MF-Block'.format(len(net_configs['MF'])))
        print('------------------Config Info--------------------\n')

    return net_configs

def MobileFormer_Mid_Config(alpha=1.0, print_info=False):
    """配置对应原文: Mobile-Former-151M
    """
    net_configs = {
            # Config Params:
            # {'base_cfg':..., 'stem':..., 'bneck-lite':..., 'MF':..., 'head':...}
            # base_cfg: MobileFormer基本配置参数
            # stem: Stem层配置参数
            # bneck-lite: Lite_BottleNeck层配置参数
            # MF: Basic_Block层配置参数
            # head: Classifier_Head层配置参数
            'base_cfg': [ # m, embed_dims, num_head, mlp_ratio,
                          # q_bias, kv_bias, qkv_bias,
                          # dropout_rate, droppath_rate,
                          # attn_dropout_rate, mlp_dropout_rate,
                          # k, use_mlp,
                          # coefs, const, dy_reduce, groups
                            6, 192, 4, 2,
                            True, True, True,
                            0., 0.,
                            0., 0.,
                            2, True,
                            [1.0, 0.5], [1.0, 0.0], 6, 1
                        ],

            'stem': [ # in_channels, out_channels, kernel_size, stride, padding, act
                        3, int(alpha*12), 3, 2, 1, nn.Hardswish
                    ],

            'bneck-lite': [ # in_channels, hidden_channels, out_channels, 
                            # expands, kernel_size, stride, padding, act
                              int(alpha*12), int(alpha*24), int(alpha*12),
                              2, 3, 1, 1, nn.ReLU6
                          ],
            
            'MF': [ # in_channels, out_channels,
                    # t, kernel_size, stride, padding,
                    # add_pointwise_conv, pointwise_conv_channels,
                    # act, norm
                      [
                          int(alpha*12), int(alpha*16),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_1 Config(downsample)
                      [
                          int(alpha*16), int(alpha*16),
                          3, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_2 Config
                      [
                          int(alpha*16), int(alpha*32),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_3 Config(downsample)
                      [
                          int(alpha*32), int(alpha*32),
                          3, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_4 Config
                      [
                          int(alpha*32), int(alpha*64),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_5 Config(downsample)
                      [
                          int(alpha*64), int(alpha*64),
                          4, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_6 Config
                      [
                          int(alpha*64), int(alpha*88),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_7 Config
                      [
                          int(alpha*88), int(alpha*88),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_8 Config
                      [
                          int(alpha*88), int(alpha*128),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_9 Config(downsample)
                      [
                          int(alpha*128), int(alpha*128),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_10 Config
                      [
                          int(alpha*128), int(alpha*128),
                          6, 3, 1, 1,
                          True, 768,
                          nn.GELU, nn.LayerNorm
                      ] # MF_11 Config
                  ],

            'head': [ # in_channels, head_type, act
                        768, 'mid', nn.Hardswish
                    ]
        }

    if print_info == True:
        parse_config(net_configs)
        print('------------------Config Info--------------------')
        print('The Create Mobile-Former Model Has -{0}- MF-Block'.format(len(net_configs['MF'])))
        print('------------------Config Info--------------------\n')

    return net_configs

def MobileFormer_Mid_Small_Config(alpha=1.0, print_info=False):
    """配置对应原文: Mobile-Former-96M
    """
    net_configs = {
            # Config Params:
            # {'base_cfg':..., 'stem':..., 'bneck-lite':..., 'MF':..., 'head':...}
            # base_cfg: MobileFormer基本配置参数
            # stem: Stem层配置参数
            # bneck-lite: Lite_BottleNeck层配置参数
            # MF: Basic_Block层配置参数
            # head: Classifier_Head层配置参数
            'base_cfg': [ # m, embed_dims, num_head, mlp_ratio,
                          # q_bias, kv_bias, qkv_bias,
                          # dropout_rate, droppath_rate,
                          # attn_dropout_rate, mlp_dropout_rate,
                          # k, use_mlp,
                          # coefs, const, dy_reduce, groups
                            4, 128, 4, 2,
                            True, True, True,
                            0., 0.,
                            0., 0.,
                            2, True,
                            [1.0, 0.5], [1.0, 0.0], 6, 1
                        ],

            'stem': [ # in_channels, out_channels, kernel_size, stride, padding, act
                        3, int(alpha*12), 3, 2, 1, nn.Hardswish
                    ],

            'bneck-lite': [ # in_channels, hidden_channels, out_channels, 
                            # expands, kernel_size, stride, padding, act
                              int(alpha*12), int(alpha*24), int(alpha*12),
                              2, 3, 1, 1, nn.ReLU6
                          ],
            
            'MF': [ # in_channels, out_channels,
                    # t, kernel_size, stride, padding,
                    # add_pointwise_conv, pointwise_conv_channels,
                    # act, norm
                      [
                          int(alpha*12), int(alpha*16),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_1 Config(downsample)
                      [
                          int(alpha*16), int(alpha*32),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_2 Config(downsample)
                      [
                          int(alpha*32), int(alpha*32),
                          3, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_3 Config
                      [
                          int(alpha*32), int(alpha*64),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_4 Config
                      [
                          int(alpha*64), int(alpha*64),
                          4, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_5 Config(downsample)
                      [
                          int(alpha*64), int(alpha*88),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_6 Config
                      [
                          int(alpha*88), int(alpha*128),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_7 Config
                      [
                          int(alpha*128), int(alpha*128),
                          6, 3, 1, 1,
                          True, 768,
                          nn.GELU, nn.LayerNorm
                      ] # MF_8 Config
                  ],

            'head': [ # in_channels, head_type, act
                        768, 'mid_small', nn.Hardswish
                    ]
        }

    if print_info == True:
        parse_config(net_configs)
        print('------------------Config Info--------------------')
        print('The Create Mobile-Former Model Has -{0}- MF-Block'.format(len(net_configs['MF'])))
        print('------------------Config Info--------------------\n')

    return net_configs

def MobileFormer_Small_Config(alpha=1.0, print_info=False):
    """配置对应原文: Mobile-Former-52M
    """
    net_configs = {
            # Config Params:
            # {'base_cfg':..., 'stem':..., 'bneck-lite':..., 'MF':..., 'head':...}
            # base_cfg: MobileFormer基本配置参数
            # stem: Stem层配置参数
            # bneck-lite: Lite_BottleNeck层配置参数
            # MF: Basic_Block层配置参数
            # head: Classifier_Head层配置参数
            'base_cfg': [ # m, embed_dims, num_head, mlp_ratio,
                          # q_bias, kv_bias, qkv_bias,
                          # dropout_rate, droppath_rate,
                          # attn_dropout_rate, mlp_dropout_rate,
                          # k, use_mlp,
                          # coefs, const, dy_reduce, groups
                            3, 128, 4, 2,
                            True, True, True,
                            0., 0.,
                            0., 0.,
                            2, True,
                            [1.0, 0.5], [1.0, 0.0], 6, 1
                        ],

            'stem': [ # in_channels, out_channels, kernel_size, stride, padding, act
                        3, int(alpha*8), 3, 2, 1, nn.Hardswish
                    ],

            'bneck-lite': [ # in_channels, hidden_channels, out_channels, 
                            # expands, kernel_size, stride, padding, act
                              int(alpha*8), int(alpha*24), int(alpha*12),
                              2, 3, 2, 1, nn.ReLU6
                          ],
            
            'MF': [ # in_channels, out_channels,
                    # t, kernel_size, stride, padding,
                    # add_pointwise_conv, pointwise_conv_channels,
                    # act, norm
                      [
                          int(alpha*12), int(alpha*12),
                          3, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_1 Config
                      [
                          int(alpha*12), int(alpha*24),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_2 Config(downsample)
                      [
                          int(alpha*24), int(alpha*24),
                          3, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_3 Config
                      [
                          int(alpha*24), int(alpha*48),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_4 Config
                      [
                          int(alpha*48), int(alpha*48),
                          4, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_5 Config(downsample)
                      [
                          int(alpha*48), int(alpha*64),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_6 Config
                      [
                          int(alpha*64), int(alpha*96),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_7 Config
                      [
                          int(alpha*96), int(alpha*96),
                          6, 3, 1, 1,
                          True, 576,
                          nn.GELU, nn.LayerNorm
                      ] # MF_8 Config
                  ],

            'head': [ # in_channels, head_type, act
                        576, 'small', nn.Hardswish
                    ]
        }

    if print_info == True:
        parse_config(net_configs)
        print('------------------Config Info--------------------')
        print('The Create Mobile-Former Model Has -{0}- MF-Block'.format(len(net_configs['MF'])))
        print('------------------Config Info--------------------\n')

    return net_configs

def MobileFormer_Tiny_Config(alpha=1.0, print_info=False):
    """配置对应原文: Mobile-Former-26M
    """
    net_configs = {
            # Config Params:
            # {'base_cfg':..., 'stem':..., 'bneck-lite':..., 'MF':..., 'head':...}
            # base_cfg: MobileFormer基本配置参数
            # stem: Stem层配置参数
            # bneck-lite: Lite_BottleNeck层配置参数
            # MF: Basic_Block层配置参数
            # head: Classifier_Head层配置参数
            'base_cfg': [ # m, embed_dims, num_head, mlp_ratio,
                          # q_bias, kv_bias, qkv_bias,
                          # dropout_rate, droppath_rate,
                          # attn_dropout_rate, mlp_dropout_rate,
                          # k, use_mlp,
                          # coefs, const, dy_reduce, groups
                            3, 128, 4, 2,
                            True, True, True,
                            0., 0.,
                            0., 0.,
                            2, True,
                            [1.0, 0.5], [1.0, 0.0], 6, 4
                        ],

            'stem': [ # in_channels, out_channels, kernel_size, stride, padding, act
                        3, int(alpha*8), 3, 2, 1, nn.Hardswish
                    ],

            'bneck-lite': [ # in_channels, hidden_channels, out_channels, 
                            # expands, kernel_size, stride, padding, act
                              int(alpha*8), int(alpha*24), int(alpha*12),
                              2, 3, 2, 1, nn.ReLU6
                          ],
            
            'MF': [ # in_channels, out_channels,
                    # t, kernel_size, stride, padding,
                    # add_pointwise_conv, pointwise_conv_channels,
                    # act, norm
                      [
                          int(alpha*12), int(alpha*12),
                          3, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_1 Config
                      [
                          int(alpha*12), int(alpha*24),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_2 Config(downsample)
                      [
                          int(alpha*24), int(alpha*24),
                          3, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_3 Config
                      [
                          int(alpha*24), int(alpha*48),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_4 Config
                      [
                          int(alpha*48), int(alpha*48),
                          4, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_5 Config(downsample)
                      [
                          int(alpha*48), int(alpha*64),
                          6, 3, 1, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_6 Config
                      [
                          int(alpha*64), int(alpha*96),
                          6, 3, 2, 1,
                          False, None,
                          nn.GELU, nn.LayerNorm
                      ], # MF_7 Config
                      [
                          int(alpha*96), int(alpha*96),
                          6, 3, 1, 1,
                          True, 576,
                          nn.GELU, nn.LayerNorm
                      ] # MF_8 Config
                  ],

            'head': [ # in_channels, head_type, act
                        576, 'small', nn.Hardswish
                    ]
        }

    if print_info == True:
        parse_config(net_configs)
        print('------------------Config Info--------------------')
        print('The Create Mobile-Former Model Has -{0}- MF-Block'.format(len(net_configs['MF'])))
        print('------------------Config Info--------------------\n')

    return net_configs

def parse_base(base_cfg):
    # token
    print("\n-------------------------------------")
    print('--Token Config--\nToken Size:\t\t\t [{0}, {1}]'.format(base_cfg[0], base_cfg[1]))
    # attention
    print("\n-------------------------------------")
    print('--Attention Config--\nNum Head:\t\t\t {0}\nMLP Ratio:\t\t\t {1}'.format(base_cfg[2], base_cfg[3]))
    # former bridge
    print("\n-------------------------------------")
    print('--Former Bridge Config--\nQuery Bias:\t\t\t {0}'.format(base_cfg[4]))
    # dy_relu
    print("\n-------------------------------------")
    print('--DY_ReLU Config--\nK:\t\t\t\t {0}\nUse MLP:\t\t\t {1}\nMLP Ratio:\t\t\t {2}'.format(base_cfg[11], base_cfg[12], base_cfg[3]))
    print('Coefs:\t\t\t\t {0}\nConst:\t\t\t\t {1}'.format(base_cfg[13], base_cfg[14]))
    print('DY Reduce:\t\t\t {0}'.format(base_cfg[15]))
    # pointwise conv
    print("\n-------------------------------------")
    print('--PointWise Conv Config--\nGroups:\t\t\t\t {0}'.format(base_cfg[16]))
    # bias
    print("\n-------------------------------------")
    print('--Bias Config--\nQuery Bias:\t\t\t {0}\nKey-Value Bias:\t\t\t {1}'.format(base_cfg[4], base_cfg[5]))
    print('Query-Key-Value Bias:\t\t {0}'.format(base_cfg[6]))
    # drop
    print("\n-------------------------------------")
    print('--Drop Config--\nDropout Rate:\t\t\t {0}'.format(base_cfg[7]))
    print('Droppath Rate:\t\t\t {0}\nAttn_Dropout Rate:\t\t {1}\nMlp_Dropout Rate:\t\t {2}'.format(base_cfg[8], base_cfg[9], base_cfg[10]))

def parse_head(head_cfg):
    print('--The Head Layer Config--')
    print('Input Channel:\t\t\t', head_cfg[0])
    print('Head Type:\t\t\t', head_cfg[1])
    print('Head ACT:\t\t\t', head_cfg[2])

def parse_stem(stem_cfg):
    print('--The Stem Layer Config--')
    print('Input Channel:\t\t\t', stem_cfg[0])
    print('Output Channel:\t\t\t', stem_cfg[1])
    print('Kernel Size:\t\t\t', stem_cfg[2])
    print('Stride:\t\t\t\t', stem_cfg[3])
    print('Padding:\t\t\t', stem_cfg[4])
    print('Act:\t\t\t', stem_cfg[5])

def parse_bneck(bneck_cfg):
    print('--The Lite BottleNeck Layer Config--')
    print('Input Channel:\t\t\t', bneck_cfg[0])
    print('Hidden Channel:\t\t\t', bneck_cfg[1])
    print('Output Channel:\t\t\t', bneck_cfg[2])
    print('Expands:\t\t\t', bneck_cfg[3])
    print('Kernel Size:\t\t\t', bneck_cfg[4])
    print('Stride:\t\t\t\t', bneck_cfg[5])
    print('Padding:\t\t\t', bneck_cfg[6])
    print('Act:\t\t\t', bneck_cfg[7])

def parse_mf(mf_cfg):
    print('--The MF Block Layer Config--')
    for idx, _cfgs in enumerate(mf_cfg):
        print("\n-------------------------------------")
        print('*Block-{0}-'.format(idx+1))
        print('T:\t\t\t\t', _cfgs[2])
        print('Input Channel:\t\t\t', _cfgs[0])
        print('Output Channel:\t\t\t', _cfgs[1])
        print('Kernel Size:\t\t\t', _cfgs[3])
        print('Stride:\t\t\t\t', _cfgs[4])
        print('Padding:\t\t\t', _cfgs[5])
        print('Add PointWise Conv:\t\t', _cfgs[6])
        print('PointWise Conv Channel:\t\t', _cfgs[7])
        print('Act:\t\t\t\t', _cfgs[8])
        print('Norm:\t\t\t\t', _cfgs[9])

def parse_config(configs):
    base_cfg = configs['base_cfg']
    stem_cfg = configs['stem']
    bneck_cfg = configs['bneck-lite']
    mf_cfg = configs['MF']
    head_cfg = configs['head']

    print("=============================================================")
    parse_base(base_cfg)
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    parse_stem(stem_cfg)
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    parse_bneck(bneck_cfg)
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    parse_mf(mf_cfg)
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    parse_head(head_cfg)
    print("=============================================================\n")

def MobileFormer_Config(model_type='base', alpha=1.0, print_info=False):
    config_names = MobileFormer_Config_Names()
    assert model_type in config_names, \
        "Error: Please choice the model_type in [{0}].".format(config_names)
    # 'big', 'base', 'base_small',
    # 'mid', 'mid_small', 'samll', 'tiny'
    if model_type == 'big':
        return MobileFormer_Big_Config(alpha=alpha, print_info=False)
    elif model_type == 'base':
        return MobileFormer_Base_Config(alpha=alpha, print_info=False)
    elif model_type == 'base_small':
        return MobileFormer_Base_Small_Config(alpha=alpha, print_info=False)
    elif model_type == 'mid':
        return MobileFormer_Mid_Config(alpha=alpha, print_info=False)
    elif model_type == 'mid_small':
        return MobileFormer_Mid_Small_Config(alpha=alpha, print_info=False)
    elif model_type == 'small':
        return MobileFormer_Small_Config(alpha=alpha, print_info=False)

    return MobileFormer_Tiny_Config(alpha=alpha, print_info=False)
