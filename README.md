# The-MobileNet-line-of-work-by-PaddlePaddle
The MobileNet line of work by PaddlePaddle

- **<font color="red">MobileNet(V1)</font>** 模型复现源码: <a href="./mobilenet.py">`mobilenet.py`</a>
    <table>
        <tr>
            <td align="center">V1网络结构</td>
            <td align="center">深度分离卷积</td>
        </tr>
        <tr>
            <td align="center"><img src="./images/v1_config.png" border=0></td>
            <td align="center"><img src="./images/v1_dp_conv.png" border=0></td>
        </tr>
    </table>
    
    - <a href="https://arxiv.org/abs/1704.04861" target="_blank">`Go to Read Paper`</a>
    - `done date`: `2021-11-12`
    - 模型结构(组网)类型:
    
        - `DepthWise_Conv`: 深度卷积--每个通道分配单个卷积核并输出，in_channels=out_channels
        - `PointWise_Conv`: 逐点卷积--1x1卷积
        - `Depth_Separ_Conv`: 深度可分离卷积--`深度卷积`+`逐点卷积`, 输出经过relu，非线性输出
        - `Stem`: 渐入层--进行第一次图像的特征提取，空间信息转换
        - `Classifier_Head`: 分类头--输出分类结果
        - `MobileNet`: 组网实现

    - 构建模型接口说明:
 
        - `MobileNet`: 构建模型的基类
        - `MobileNet_Base`: 构建**基础**MobileNet的函数
        - `MobileNet_0_75`: 构建**较大**MobileNet的函数
        - `MobileNet_Mid`: 构建**中等**MobileNet的函数
        - `MobileNet_Small`: 构建**最小**MobileNet的函数

- **<font color="red">MobileNetV2</font>** 模型复现源码: <a href="./mobilenetV2.py">`mobilenetV2.py`</a>
    <table>
        <tr>
            <td align="center">V2网络结构</td>
            <td align="center">倒残差结构</td>
        </tr>
        <tr>
            <td align="center"><img src="./images/v2_config.png" border=0></td>
            <td align="center"><img src="./images/v2_res.png" border=0></td>
        </tr>
    </table>
    
    - <a href="https://arxiv.org/abs/1801.04381" target="_blank">`Go to Read Paper`</a>
    - `done date`: `2021-11-12`
    - 模型结构(组网)类型:
    
        - `DepthWise_Conv`: 深度卷积--每个通道分配单个卷积核并输出，in_channels=out_channels
        - `PointWise_Conv`: 逐点卷积--1x1卷积
        - `Linear_Depth_Separ_Conv`: 深度可分离卷积--`深度卷积`+`逐点卷积`, 输出不经过relu等，线性输出
        - `BottleNeck`: 倒残差模块--低密度空间残差
        - `Stem`: 渐入层--进行第一次图像的特征提取，空间信息转换
        - `Classifier_Head`: 分类头--输出分类结果
        - `MobileNetV2`: 组网实现
    
    - 构建模型接口说明:
 
        - `MobileNetV2`: 构建模型的基类
        - `MobileNetV2_for_224`: 构建**最适合224大小图像**的MobileNetV2的函数
        - `MobileNetV2_Base`: 构建**基础**MobileNetV2的函数
        - `MobileNetV2_0_75`: 构建**较大**MobileNetV2的函数
        - `MobileNetV2_Mid`: 构建**中等**MobileNetV2的函数
        - `MobileNetV2_Small`: 构建**最小**MobileNetV2的函数

- **<font color="red">MobileNetV3</font>** 模型复现源码: <a href="./mobilenetV3.py">`mobilenetV3.py`</a>
    <table>
        <tr>
            <td align="center">V3激活函数</td>
            <td align="center">V3注意力SE模块</td>
        </tr>
        <tr>
            <td align="center"><img src="./images/v3_act.png" border=0></td>
            <td align="center"><img src="./images/se_attention.png" border=0></td>
        </tr>
    <tr>
            <td align="center">V3-Large结构</td>
            <td align="center">V3-Small结构</td>
        </tr>
        <tr>
            <td align="center"><img src="./images/v3_large_config.png" border=0></td>
            <td align="center"><img src="./images/v3_small_config.png" border=0></td>
        </tr>
    </table>
    
    - <a href="https://arxiv.org/abs/1905.02244" target="_blank">`Go to Read Paper`</a>
    - `done date`: `2021-11-12`
    - 模型结构(组网)类型:
    
        - `Identify`: 占位符--不做任何操作, x=f(x)
        - `DepthWise_Conv`: 深度卷积--每个通道分配单个卷积核并输出，in_channels=out_channels
        - `PointWise_Conv`: 逐点卷积--1x1卷积
        - `Linear_Depth_Separ_Conv`: 深度可分离卷积--`深度卷积`+`逐点卷积`, 输出不经过relu等，线性输出
        - `SEBlock`: SE Attention计算模块--channels注意力
        - `BottleNeck`: 倒残差模块--低密度空间残差
        - `Stem`: 渐入层--进行第一次图像的特征提取，空间信息转换
        - `Classifier_Head`: 分类头--输出分类结果
        - `MobileNetV3_Large`: Large结构的组网实现
        - `MobileNetV3_Small`: Small结构的组网实现
    
    - 构建模型接口说明:
 
        - `MobileNetV3_Large`: 构建Large模型的基类
        - `MobileNetV3_Large_for_224`: 构建**最适合224大小图像**的MobileNetV3_Large的函数
        - `MobileNetV3_Large_Base`: 构建**基础**MobileNetV3_Large的函数
        - `MobileNetV3_Large_0_75`: 构建**较大**MobileNetV3_Large的函数
        - `MobileNetV3_Large_Mid`: 构建**中等**MobileNetV3_Large的函数
        - `MobileNetV3_Large_Small`: 构建**最小**MobileNetV3_Large的函数
        - `MobileNetV3_Small`: 构建Small模型的基类
        - `MobileNetV3_Small_for_224`: 构建**最适合224大小图像**的MobileNetV3_Small的函数
        - `MobileNetV3_Small_Base`: 构建**基础**MobileNetV3_Small的函数
        - `MobileNetV3_Small_0_75`: 构建**较大**MobileNetV3_Small的函数
        - `MobileNetV3_Small_Mid`: 构建**中等**MobileNetV3_Small的函数
        - `MobileNetV3_Small_Small`: 构建**最小**MobileNetV3_Small的函数

- **<font color="red">Mobile-Former</font>** 模型复现源码: <a href="./mobilenet.py">`mobile-former.py`</a>
    <table>
        <tr>
            <td align="center">MF激活函数</td>
            <td align="center">MF性能</td>
        </tr>
        <tr>
            <td align="center"><img src="./images/mf_dy_relu.png" border=0></td>
            <td align="center"><img src="./images/mf_performance.png" border=0></td>
        </tr>
    <tr>
            <td align="center">MF架构</td>
            <td align="center">MF组网配置</td>
        </tr>
        <tr>
            <td align="center"><img src="./images/mf_arch.png" border=0></td>
            <td align="center"><img src="./images/mf_config.png" border=0></td>
        </tr>
    </table>
    
    - <a href="https://arxiv.org/abs/2108.05895" target="_blank">`Go to Read Paper`</a>
    - `done date`: `2021-11-13`
    - 模型结构(组网)类型:
 
        - `DepthWiseConv`: 深度卷积--每个通道分配单个卷积核并输出，in_channels=out_channels
        - `PointWiseConv`: 逐点卷积--1x1卷积
        - `MLP`: 多层感知机
        - `DyReLU`: 动态ReLU
        - `Mobile`: MF中的Mobile卷积部分
        - `Attention`: 简单的普通注意力机制
        - `DropPath`: Path丢弃
        - `Former`: MF中的Former纯注意力部分
        - `ToFormer_Bridge`: 从Mobile到Former的桥
        - `ToMobile_Bridge`: 从Mobile到Former的桥
        - `Stem`: 渐入层
        - `BottleNeck`: BottleNeck -- 支持Lite类型
        - `Classifier_Head`: 分类头
        - `MFBlock`: MobileFormer最小实现单元
        - `MobileFormer`: 网络实现
    
    - 构建模型接口说明:
    
        - `MobileFormer`: 构建MobileFormer模型的基类
            - 传入`model_type`: [`26m`, `52m`, `96m`, `151m`, `214m`, `294m`, `508m`],
            - 生成对应的模型
        - `check_model_size`: 检查模型参数量对比情况(与原论文)
    
    - 参数对齐情况:
        ```python
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
        ```



