## 介绍
本项目上是为了测试win10上基于onnx模型，用tensorrt c++ API推理 不同batch的对比情况，包括数据预处理，推理时间等
## 环境
*  vs2017, win10，tensorrt7.0 ,cuda10.0,cudnn7.6.3

## 编译运行
* 参数看`resnet_main.cpp`说明

## 其他说明
* 本工程时参考[项目](https://github.com/Syencil/tensorRT)，由于其时在Linux环江编译，将其移植到window，做了一些修改，增加batch推理，需要onnx model也修改。