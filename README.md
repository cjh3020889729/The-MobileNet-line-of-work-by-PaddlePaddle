# The-MobileNet-line-of-work-by-PaddlePaddle
The MobileNet line of work by PaddlePaddle

- **<font color="red">MobileNet(V1)</font>** 模型复现源码: <a href="./mobilenet.py">`mobilenet.py`</a>
    <table><tr>
        <td><font text-align="center">V1网络结构</font><img src="./images/v1_config.png" border=0></td>
        <td><font text-align="center">深度分离卷积</font><img src="./images/v1_dp_conv.png" border=0></td>
    </tr></table>
    
    - `done date`: `2021-11-12`
    - 构建模型接口说明:
 
        - `MobileNet`: 构建模型的基类
        - `MobileNet_Base`: 构建**基础**MobileNet的函数
        - `MobileNet_0_75`: 构建**较大**MobileNet的函数
        - `MobileNet_Mid`: 构建**中等**MobileNet的函数
        - `MobileNet_Small`: 构建**最小**MobileNet的函数

- **<font color="red">MobileNetV2</font>** 模型复现源码: <a href="./mobilenetV2.py">`mobilenetV2.py`</a>
    <table><tr>
        <td><img src="./images/v2_config.png" border=0></td>
        <td><img src="./images/v2_res.png" border=0></td>
    </tr></table>
    
    - `done date`: `2021-11-12`
    - 构建模型接口说明:
 
        - `MobileNetV2`: 构建模型的基类
        - `MobileNetV2_for_224`: 构建**最适合224大小图像**的MobileNetV2的函数
        - `MobileNetV2_Base`: 构建**基础**MobileNetV2的函数
        - `MobileNetV2_0_75`: 构建**较大**MobileNetV2的函数
        - `MobileNetV2_Mid`: 构建**中等**MobileNetV2的函数
        - `MobileNetV2_Small`: 构建**最小**MobileNetV2的函数

- **<font color="red">MobileNetV3</font>** 模型复现源码: `mobilenetV3.py` -- To Do
    <table><tr>
        <td><img src="./images/v3_act.png" border=0></td>
        <td><img src="./images/se_attention.png" border=0></td>
    </tr></table>
    <table><tr>
        <td><img src="./images/v3_large_config.png" border=0></td>
        <td><img src="./images/v3_small_config.png" border=0></td>
    </tr></table>


- **<font color="red">Mobile-Former</font>** 模型复现源码: `mobileformer.py` -- To Do
