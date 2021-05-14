// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/6

#include "resnet.h"

Resnet::Resnet(common::InputParams inputParams, common::TrtParams trtParams, common::ClassificationParams classifactionParams) : TensorRT(std::move(inputParams), std::move(trtParams)), mClassifactionParams(std::move(classifactionParams)){

}

std::vector<float> Resnet::preProcess(const std::vector<cv::Mat> &images) const {
	std::vector<float> fileData = imagePreprocess(images, mInputParams.ImgH, mInputParams.ImgW, mInputParams.IsPadding, mInputParams.pFunction, false,mInputParams.workers);
	return fileData;
}
std::vector<float> Resnet::preProcess_batch(const std::vector<std::vector<float>> &images) const {
	//const auto start_t = std::chrono::high_resolution_clock::now();
	//std::vector<float> fileData = imagePreprocess(images, mInputParams.ImgH, mInputParams.ImgW, mInputParams.IsPadding, mInputParams.pFunction, false, mInputParams.workers);
	std::vector<float> fileData = imagePreprocess2batch(images, mInputParams.ImgH, mInputParams.ImgW);
	//const auto end_t = std::chrono::high_resolution_clock::now();
	//std::cout<<"preprocess image cost:"<<std::chrono::duration<double, std::milli>(end_t - start_t).count() << "ms" << std::endl;
	return fileData;
}


std::vector<float> Resnet::postProcess(common::BufferManager &bufferManager) const {
    assert(mInputParams.OutputTensorNames.size()==1);
    std::vector<float> prob(mClassifactionParams.NumClass*mInputParams.BatchSize, 0);
    auto *origin_output = static_cast<const float*>(bufferManager.getHostBuffer(mInputParams.OutputTensorNames[0]));
    for(int i=0; i<mClassifactionParams.NumClass*mInputParams.BatchSize; ++i){
        prob[i] = origin_output[i];
    }
    return prob;
}

bool Resnet::initSession(int initOrder) {
    return TensorRT::initSession(initOrder);
}

std::vector<float> Resnet::predOneImage(const std::vector<cv::Mat> &imagevector) {
    assert(mInputParams.BatchSize==1);
    common::BufferManager bufferManager(mCudaEngine, 1);
	float elapsedTime = infer(std::vector<std::vector<float>>{preProcess(imagevector)}, bufferManager, nullptr);
    gLogInfo << "Infer time is "<< elapsedTime << "ms" << std::endl;
    std::vector<float> prob = postProcess(bufferManager);
    return prob;
}
std::vector<float> Resnet::batchImage(const std::vector<std::vector<float>> &imagevector) {
	//assert(mInputParams.BatchSize==1);
	common::BufferManager bufferManager(mCudaEngine, mInputParams.BatchSize);
	float elapsedTime = infer(std::vector<std::vector<float>>{preProcess_batch(imagevector)}, bufferManager, nullptr);
	gLogInfo << "Infer time is " << elapsedTime << "ms" << std::endl;
	std::vector<float> prob = postProcess(bufferManager);
	return prob;
}

