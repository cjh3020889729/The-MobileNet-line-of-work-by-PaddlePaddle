# The-MobileNet-line-of-work-by-PaddlePaddle
The MobileNet line of work by PaddlePaddle

- MobileNet(V1)模型复现源码: `mobilenet.py`

    - `done time`: `2021-11-12`
    - 构建模型接口说明:
 
        - `MobileNet`: 构建模型的基类
        - `MobileNet_Base`: 构建基础MobileNet的函数
        - `MobileNet_0_75`: 构建较大MobileNet的函数
        - `MobileNet_Mid`: 构建中等MobileNet的函数
        - `MobileNet_Small`: 构建最小MobileNet的函数

- MobileNetV2模型复现源码: `mobilenetV2.py`

    - `done time`: `2021-11-12`
    - 构建模型接口说明:
 
        - `MobileNetV2`: 构建模型的基类
        - `MobileNetV2_for_224`: 构建最适合224大小图像的MobileNetV2的函数
        - `MobileNetV2_Base`: 构建基础MobileNetV2的函数
        - `MobileNetV2_0_75`: 构建较大MobileNetV2的函数
        - `MobileNetV2_Mid`: 构建中等MobileNetV2的函数
        - `MobileNetV2_Small`: 构建最小MobileNetV2的函数

- MobileNetV3模型复现源码: `mobilenetV3.py` -- To Do

- Mobile-Former模型复现源码: `mobileformer.py` -- To Do
