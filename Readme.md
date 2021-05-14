## 介绍
本项目上是为了测试win10上基于onnx模型，用tensorrt c++ API推理 不同batch的对比情况，包括数据预处理，推理时间等
## 环境
*  vs2017, win10，tensorrt7.0 ,cuda10.0,cudnn7.6.3

## 编译运行
* 参数看`resnet_main.cpp`说明

## 推理速度上优化中的问题   
 
* 1. 打印整个模型的各部分的速度结果  
    
    |图片解码|预处理|推理|总耗时|
    |----|----|----|----|
    |50ms|52ms|2.2ms|106ms|
> **Note** 所有的预处理都是基于opencv 在CPU执行。

其中推理fp32大约耗时2.2，int8推理耗时1.1ms。  
图片解码和预处理耗时较大，重点优化解码和预处理。
* 2. 使用nvjpeg推理结果：  
    
|图片解码和resize|预处理（归一化等）|推理|总耗时|
|----|----|----|----|
|10ms|22ms|1.1ms|34ms|

>**note:** 使用nvjpeg进行jpeg decode和resize操作，返回的还是`cv::Mat`然后再进行mean和scale以及`chw`与`hwc`操作.推理时间的减少是因为使用了batch（16个batch）推理方案。
* 3. 与onnxruntime推理对比结果  
    
|图片解码|预处理|推理|总耗时|
|----|----|----|----|
|33ms|0.7ms|21.3ms|58ms|
   
* 4. 重写归一化和nchw转化后推理结果  
    
|图片解码及预处理|batch排列|推理|总耗时|
|----|----|----|----|
|14.1ms|3.6ms|0.91ms|18ms|

> **Note** 测试中以batch为16  
    
其中，图片解码和预处理包括，图像解码操作，resize，归一化，nchw排列。


## 其他说明
* 本工程时参考[项目](https://github.com/Syencil/tensorRT)，由于其时在Linux环江编译，将其移植到window，做了一些修改，增加batch推理，需要onnx model也修改。