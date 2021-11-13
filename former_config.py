import paddle
from paddle import nn


def MobileFormer_Config_Names():
    names_ = [
                'big', 'base', 'base_small',
                'mid', 'mid_small', 'samll',
                'tiny'
             ]
    return names_

def MobileFormer_Big_Config(alpha=1.0):
    pass

def MobileFormer_Base_Config(alpha=1.0):
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
                          # coefs, const, dy_reduce
                            6, 192, 8, 2,
                            True, True, True,
                            0., 0.,
                            0., 0.,
                            2, True,
                            [1.0, 0.5], [1.0, 0.0], 6
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

    print('------------------Config Info--------------------')
    print('The Create Mobile-Former Model Has -{0}- MF-Block'.format(len(net_configs['MF'])))
    print('------------------Config Info--------------------\n')

    return net_configs

def MobileFormer_Base_Small_Config(alpha=1.0):
    pass

def MobileFormer_Mid_Config(alpha=1.0):
    pass

def MobileFormer_Mid_Small_Config(alpha=1.0):
    pass

def MobileFormer_Small_Config(alpha=1.0):
    pass

def MobileFormer_Tiny_Config(alpha=1.0):
    pass

def MobileFormer_Config(model_type='base', alpha=1.0):
    config_names = MobileFormer_Config_Names()
    assert model_type in config_names, \
        "Error: Please choice the model_type in [{0}].".format(config_names)
    
    # 'big', 'base', 'base_small',
    # 'mid', 'mid_small', 'samll', 'tiny'
    if model_type == 'big':
        return MobileFormer_Big_Config(alpha=alpha)
    elif model_type == 'base':
        return MobileFormer_Base_Config(alpha=alpha)
    elif model_type == 'base_small':
        return MobileFormer_Base_Small_Config(alpha=alpha)
    elif model_type == 'mid':
        return MobileFormer_Mid_Config(alpha=alpha)
    elif model_type == 'mid_small':
        return MobileFormer_Mid_Small_Config(alpha=alpha)
    elif model_type == 'samll':
        return MobileFormer_Small_Config(alpha=alpha)

    return MobileFormer_Tiny_Config(alpha=alpha)
