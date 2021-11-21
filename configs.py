def Config_508M(in_channels=3):
    net_config = [
                    # Stem Params Info
                    # in_channels, out_channels,
                    # kernel_size, stride, padding
                    [in_channels, 24, 3, 2, 1],
                    # Bneck-Lite Params Info
                    # in_channels, hidden_channels, out_channels,
                    # kernel_size, stride, padding
                    [24, 48, 24, 3, 1, 1],
                    # 294M-Model MFBlock Params Info
                    # in_channels, hidden_channels, out_channels,
                    # kernel_size, stride, padding
                    [24, 144, 40, 3, 2, 1], # 1 block config
                    [40, 120, 40, 3, 1, 1], # 2 block config
                    [40, 240, 72, 3, 2, 1], # 3 block config
                    [72, 216, 72, 3, 1, 1], # 4 block config
                    [72, 432, 128, 3, 2, 1], # 5 block config
                    [128, 512, 128, 3, 1, 1], # 6 block config
                    [128, 768, 176, 3, 1, 1], # 7 block config
                    [176, 1056, 176, 3, 1, 1], # 8 block config
                    [176, 1056, 240, 3, 2, 1], # 9 block config
                    [240, 1440, 240, 3, 1, 1], # 10 block config
                    [240, 1440, 240, 3, 1, 1], # 11 block config
                    # Conv1x1 Params Info
                    # in_channels, out_channels,
                    [240, 1440],
                    # Classifier Head Params Info
                    # in_channels, hidden_features
                    [1440, 1920]
                ]
    token_config = [6, 192] # token config
    reduce = 3.2            # dyrelu config
    groups = 1              # pointwise_conv groups config
    num_head = 8            # attention head config
    return net_config, token_config, reduce, num_head, groups


def Config_294M(in_channels=3):
    net_config = [
                    # Stem Params Info
                    # in_channels, out_channels,
                    # kernel_size, stride, padding
                    [in_channels, 16, 3, 2, 1],
                    # Bneck-Lite Params Info
                    # in_channels, hidden_channels, out_channels,
                    # kernel_size, stride, padding
                    [16, 32, 16, 3, 1, 1],
                    # 294M-Model MFBlock Params Info
                    # in_channels, hidden_channels, out_channels,
                    # kernel_size, stride, padding
                    [16, 96, 24, 3, 2, 1], # 1 block config
                    [24, 96, 24, 3, 1, 1], # 2 block config
                    [24, 144, 48, 3, 2, 1], # 3 block config
                    [48, 192, 48, 3, 1, 1], # 4 block config
                    [48, 288, 96, 3, 2, 1], # 5 block config
                    [96, 384, 96, 3, 1, 1], # 6 block config
                    [96, 576, 128, 3, 1, 1], # 7 block config
                    [128, 768, 128, 3, 1, 1], # 8 block config
                    [128, 768, 192, 3, 2, 1], # 9 block config
                    [192, 1152, 192, 3, 1, 1], # 10 block config
                    [192, 1152, 192, 3, 1, 1], # 11 block config
                    # Conv1x1 Params Info
                    # in_channels, out_channels,
                    [192, 1152],
                    # Classifier Head Params Info
                    # in_channels, hidden_features
                    [1152, 1920]
                ]
    token_config = [6, 192] # token config
    reduce = 3.65           # dyrelu config
    groups = 1              # pointwise_conv groups config
    num_head = 8            # attention head config
    return net_config, token_config, reduce, num_head, groups


def Config_214M(in_channels=3):
    net_config = [
                    # Stem Params Info
                    # in_channels, out_channels,
                    # kernel_size, stride, padding
                    [in_channels, 12, 3, 2, 1],
                    # Bneck-Lite Params Info
                    # in_channels, hidden_channels, out_channels,
                    # kernel_size, stride, padding
                    [12, 24, 12, 3, 1, 1],
                    # 294M-Model MFBlock Params Info
                    # in_channels, hidden_channels, out_channels,
                    # kernel_size, stride, padding
                    [12, 72, 20, 3, 2, 1], # 1 block config
                    [20, 60, 20, 3, 1, 1], # 2 block config
                    [20, 120, 40, 3, 2, 1], # 3 block config
                    [40, 160, 40, 3, 1, 1], # 4 block config
                    [40, 240, 80, 3, 2, 1], # 5 block config
                    [80, 320, 80, 3, 1, 1], # 6 block config
                    [80, 480, 112, 3, 1, 1], # 7 block config
                    [112, 672, 112, 3, 1, 1], # 8 block config
                    [112, 672, 160, 3, 2, 1], # 9 block config
                    [160, 960, 160, 3, 1, 1], # 10 block config
                    [160, 960, 160, 3, 1, 1], # 11 block config
                    # Conv1x1 Params Info
                    # in_channels, out_channels,
                    [160, 960],
                    # Classifier Head Params Info
                    # in_channels, hidden_features
                    [960, 1600]
                ]
    token_config = [6, 192] # token config
    reduce = 4.4            # dyrelu config
    groups = 1              # pointwise_conv groups config
    num_head = 4            # attention head config
    return net_config, token_config, reduce, num_head, groups


def Config_151M(in_channels=3):
    net_config = [
                    # Stem Params Info
                    # in_channels, out_channels,
                    # kernel_size, stride, padding
                    [in_channels, 12, 3, 2, 1],
                    # Bneck-Lite Params Info
                    # in_channels, hidden_channels, out_channels,
                    # kernel_size, stride, padding
                    [12, 24, 12, 3, 1, 1],
                    # 294M-Model MFBlock Params Info
                    # in_channels, hidden_channels, out_channels,
                    # kernel_size, stride, padding
                    [12, 72, 16, 3, 2, 1], # 1 block config
                    [16, 48, 16, 3, 1, 1], # 2 block config
                    [16, 96, 32, 3, 2, 1], # 3 block config
                    [32, 96, 32, 3, 1, 1], # 4 block config
                    [32, 192, 64, 3, 2, 1], # 5 block config
                    [64, 256, 64, 3, 1, 1], # 6 block config
                    [64, 384, 88, 3, 1, 1], # 7 block config
                    [88, 528, 88, 3, 1, 1], # 8 block config
                    [88, 528, 128, 3, 2, 1], # 9 block config
                    [128, 768, 128, 3, 1, 1], # 10 block config
                    [128, 768, 128, 3, 1, 1], # 11 block config
                    # Conv1x1 Params Info
                    # in_channels, out_channels,
                    [128, 768],
                    # Classifier Head Params Info
                    # in_channels, hidden_features
                    [768, 1280]
                ]
    token_config = [6, 192] # token config
    reduce = 5.2            # dyrelu config
    groups = 1              # pointwise_conv groups config
    num_head = 4            # attention head config
    return net_config, token_config, reduce, num_head, groups


def Config_96M(in_channels=3):
    net_config = [
                    # Stem Params Info
                    # in_channels, out_channels,
                    # kernel_size, stride, padding
                    [in_channels, 12, 3, 2, 1],
                    # Bneck-Lite Params Info
                    # in_channels, hidden_channels, out_channels,
                    # kernel_size, stride, padding
                    [12, 24, 12, 3, 1, 1],
                    # 294M-Model MFBlock Params Info
                    # in_channels, hidden_channels, out_channels,
                    # kernel_size, stride, padding
                    [12, 72, 16, 3, 2, 1], # 1 block config
                    [16, 96, 32, 3, 2, 1], # 2 block config
                    [32, 96, 32, 3, 1, 1], # 3 block config
                    [32, 192, 64, 3, 2, 1], # 4 block config
                    [64, 256, 64, 3, 1, 1], # 5 block config
                    [64, 384, 88, 3, 1, 1], # 6 block config
                    [88, 528, 128, 3, 2, 1], # 7 block config
                    [128, 768, 128, 3, 1, 1], # 8 block config
                    # Conv1x1 Params Info
                    # in_channels, out_channels,
                    [128, 768],
                    # Classifier Head Params Info
                    # in_channels, hidden_features
                    [768, 1280]
                ]
    token_config = [4, 128] # token config
    reduce = 3.6            # dyrelu config
    groups = 1              # pointwise_conv groups config
    num_head = 4            # attention head config
    return net_config, token_config, reduce, num_head, groups


def Config_52M(in_channels=3):
    net_config = [
                    # Stem Params Info
                    # in_channels, out_channels,
                    # kernel_size, stride, padding
                    [in_channels, 8, 3, 2, 1],
                    # Bneck-Lite Params Info
                    # in_channels, hidden_channels, out_channels,
                    # kernel_size, stride, padding
                    [8, 24, 12, 3, 2, 1],
                    # 294M-Model Params Info
                    # in_channels, hidden_channels, out_channels,
                    # kernel_size, stride, padding
                    [12, 36, 12, 3, 1, 1], # 1 block config
                    [12, 72, 24, 3, 2, 1], # 2 block config
                    [24, 72, 24, 3, 1, 1], # 3 block config
                    [24, 144, 48, 3, 2, 1], # 4 block config
                    [48, 192, 48, 3, 1, 1], # 5 block config
                    [48, 288, 64, 3, 1, 1], # 6 block config
                    [64, 384, 96, 3, 2, 1], # 7 block config
                    [96, 576, 96, 3, 1, 1], # 8 block config
                    # Conv1x1 Params Info
                    # in_channels, out_channels,
                    [96, 576],
                    # Classifier Head Params Info
                    # in_channels, hidden_features
                    [576, 1024]
                ]
    token_config = [3, 128] # token config
    reduce = 4.2            # dyrelu config
    groups = 1              # pointwise_conv groups config
    num_head = 4            # attention head config
    return net_config, token_config, reduce, num_head, groups


def Config_26M(in_channels=3):
    net_config = [
                    # Stem Params Info
                    # in_channels, out_channels,
                    # kernel_size, stride, padding
                    [in_channels, 8, 3, 2, 1],
                    # Bneck-Lite Params Info
                    # in_channels, hidden_channels, out_channels,
                    # kernel_size, stride, padding
                    [8, 24, 12, 3, 2, 1],
                    # 294M-Model Params Info
                    # in_channels, hidden_channels, out_channels,
                    # kernel_size, stride, padding
                    [12, 36, 12, 3, 1, 1], # 1 block config
                    [12, 72, 24, 3, 2, 1], # 2 block config
                    [24, 72, 24, 3, 1, 1], # 3 block config
                    [24, 144, 48, 3, 2, 1], # 4 block config
                    [48, 192, 48, 3, 1, 1], # 5 block config
                    [48, 288, 64, 3, 1, 1], # 6 block config
                    [64, 384, 96, 3, 2, 1], # 7 block config
                    [96, 576, 96, 3, 1, 1], # 8 block config
                    # Conv1x1 Params Info
                    # in_channels, out_channels,
                    [96, 576],
                    # Classifier Head Params Info
                    # in_channels, hidden_features
                    [576, 1024]
                ]
    token_config = [3, 128] # token config
    reduce = 6              # dyrelu config
    groups = 4              # pointwise_conv groups config
    num_head = 4            # attention head config
    return net_config, token_config, reduce, num_head, groups


def update_config(in_channels, model_type='26m'):
    if model_type == '508m':
        return Config_508M(in_channels=in_channels)
    elif model_type == '52m':
        return Config_52M(in_channels=in_channels)
    elif model_type == '96m':
        return Config_96M(in_channels=in_channels)
    elif model_type == '151m':
        return Config_151M(in_channels=in_channels)
    elif model_type == '214m':
        return Config_214M(in_channels=in_channels)
    elif model_type == '294m':
        return Config_294M(in_channels=in_channels)
    
    return Config_26M(in_channels=in_channels)

